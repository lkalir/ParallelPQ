#include "ppq/gpu_verifier.h"

#include <algorithm>
#include <chrono>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "ppq/device_batch.h"
#include "ppq/hip_check.h"
#include "ppq/ntt_kernels.h"
#include "ppq/stream.h"
#include "ppq/verify_kernels.h"

namespace ppq {

/// Device buffers kept across verify() calls, so repeated runs measure the steady state
struct GpuVerifier::State {
    int param_id = 0;
    std::size_t capacity = 0;
    std::optional<KeyCacheBuffers> cache;
    std::optional<SigWorkBuffers> work;
    Stream stream;

    void ensure(const ParamSet &p, std::size_t chunk) {
        if (param_id == p.id && capacity >= chunk) {
            return;
        }
        // Release before allocating, so peak memory is one set of buffers
        cache.reset();
        work.reset();
        cache.emplace(p, chunk);
        work.emplace(p, chunk);
        param_id = p.id;
        capacity = chunk;
    }
};

namespace {

/// Times stages and sums them per name: stream work between a pair of events, host work
/// (allocating and copying with synchronous calls, which are not on the stream) by wall clock
class StageTimer {
  public:
    explicit StageTimer(hipStream_t stream) : _stream(stream) {}

    template <class Fn> void run(const char *name, Fn &&fn) {
        _marks.emplace_back(name, std::make_unique<Event>(), std::make_unique<Event>());
        std::get<1>(_marks.back())->record(_stream);
        fn();
        std::get<2>(_marks.back())->record(_stream);
    }

    template <class Fn> void host(const char *name, Fn &&fn) {
        const auto start = std::chrono::steady_clock::now();
        fn();
        _host.push_back({name, std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - start)
                                   .count()});
    }

    /// Waits for the stream, then adds this chunk's times into `totals`
    void collect(std::vector<StageTime> &totals) {
        HIP_CHECK(hipStreamSynchronize(_stream));
        for (const StageTime &h : _host) {
            add(totals, h.name, h.ms);
        }
        for (const auto &[name, start, stop] : _marks) {
            add(totals, name, elapsed_ms(*start, *stop));
        }
        _marks.clear();
        _host.clear();
    }

  private:
    static void add(std::vector<StageTime> &totals, const std::string &name, double ms) {
        for (StageTime &t : totals) {
            if (t.name == name) {
                t.ms += ms;
                return;
            }
        }
        totals.push_back({name, ms});
    }

    hipStream_t _stream;
    std::vector<std::tuple<std::string, std::unique_ptr<Event>, std::unique_ptr<Event>>> _marks;
    std::vector<StageTime> _host;
};

/// The keys a chunk uses, in first-use order, and the chunk's key indices remapped onto them
struct ChunkKeys {
    std::vector<std::uint8_t> pks;
    std::vector<std::uint32_t> key_index;
    std::size_t count = 0;
};

ChunkKeys chunk_keys(const Workload &w, std::size_t first, std::size_t count) {
    ChunkKeys ck;
    std::unordered_map<std::uint32_t, std::uint32_t> local;
    ck.key_index.reserve(count);
    for (std::size_t i = first; i < first + count; ++i) {
        const std::uint32_t k = w.key_index[i];
        const auto [it, added] = local.try_emplace(k, static_cast<std::uint32_t>(ck.count));
        if (added) {
            const ByteView pk = w.pk(k);
            ck.pks.insert(ck.pks.end(), pk.begin(), pk.end());
            ++ck.count;
        }
        ck.key_index.push_back(it->second);
    }
    return ck;
}

} // namespace

GpuVerifier::GpuVerifier(const GpuVerifierOptions &options)
    : _options(options), _state(std::make_unique<State>()) {
    if (!valid_ntt_group_size(options.ntt_group)) {
        throw std::invalid_argument("NTT group must be 16, 32, 64 or 128 threads");
    }
    if (options.threads == 0 || options.threads % options.ntt_group != 0) {
        throw std::invalid_argument("threads per block (" + std::to_string(options.threads) +
                                    ") must be a positive multiple of the NTT group (" +
                                    std::to_string(options.ntt_group) + ")");
    }
    const int max_threads = verify_kernels_max_threads();
    if (options.threads > static_cast<unsigned>(max_threads) ||
        static_cast<int>(options.threads) > ntt_coop_max_threads(options.ntt_group)) {
        throw std::invalid_argument("the stage kernels run at most " + std::to_string(max_threads) +
                                    " threads per block on this device");
    }
    if (options.chunk == 0) {
        throw std::invalid_argument("chunk size must be at least 1");
    }
}

GpuVerifier::~GpuVerifier() = default;

std::string GpuVerifier::describe() const {
    int device = 0;
    hipDeviceProp_t props{};
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    return std::string("GPU: ") + props.name + ", staged kernels, " +
           std::to_string(_options.threads) + " threads/block, NTT group " +
           std::to_string(_options.ntt_group) + ", chunks of " + std::to_string(_options.chunk);
}

VerifyResult GpuVerifier::verify(const Workload &w) {
    const auto start = std::chrono::steady_clock::now();
    const ParamSet &p = w.params;
    const std::size_t n = w.sig_count();
    const std::size_t chunk = std::min(_options.chunk, std::max<std::size_t>(n, 1));
    const unsigned t = _options.threads;
    const unsigned g = _options.ntt_group;

    VerifyResult result;
    result.valid.resize(n);
    result.verdicts.resize(n);
    if (n == 0) {
        return result;
    }

    State &s = *_state;
    s.ensure(p, chunk);
    const hipStream_t stream = s.stream.get();
    const SigWork work = s.work->view();
    StageTimer timer(stream);

    // When every key fits the cache, prepare the whole key table once and use the workload's
    // own indices; otherwise each chunk prepares just the keys it uses
    const bool shared_keys = w.key_count() <= s.capacity;
    std::optional<DeviceKeys> all_keys;
    if (shared_keys) {
        timer.host("upload", [&] { all_keys.emplace(w.pks, w.key_count()); });
        timer.run("keys", [&] {
            launch_prepare_keys(p, all_keys->view(), s.cache->view(w.key_count()), t, stream);
        });
    }

    for (std::size_t first = 0; first < n; first += chunk) {
        const std::size_t count = std::min(chunk, n - first);

        std::optional<ChunkKeys> ck;
        std::optional<DeviceKeys> keys;
        std::size_t key_count = w.key_count();
        if (!shared_keys) {
            ck = chunk_keys(w, first, count);
            key_count = ck->count;
            timer.host("upload", [&] { keys.emplace(ck->pks, ck->count); });
            timer.run("keys", [&] {
                launch_prepare_keys(p, keys->view(), s.cache->view(ck->count), t, stream);
            });
        }
        const KeyCacheView cache = s.cache->view(key_count);

        std::optional<DeviceSigBatch> batch;
        timer.host("upload",
                   [&] { batch.emplace(w, first, count, ck ? &ck->key_index : nullptr); });
        const SigInputs in = batch->view();

        timer.run("decode", [&] { launch_decode(p, in, work, t, stream); });
        timer.run("mu", [&] { launch_mu(p, in, cache, work, t, stream); });
        timer.run("challenge", [&] { launch_challenge(p, in, work, t, stream); });
        timer.run("ntt_z", [&] { launch_z_ntt(p, in, work, g, t, stream); });
        timer.run("matvec", [&] { launch_matvec(p, in, cache, work, t, stream); });
        timer.run("intt_w", [&] { launch_w_intt(p, in, work, g, t, stream); });
        timer.run("use_hint", [&] { launch_use_hint(p, in, work, t, stream); });
        timer.run("final_hash", [&] { launch_final_hash(p, in, work, t, stream); });

        std::vector<std::uint32_t> status(count);
        timer.run("download", [&] {
            HIP_CHECK(hipMemcpyAsync(status.data(), work.status, count * sizeof(std::uint32_t),
                                     hipMemcpyDeviceToHost, stream));
        });
        timer.collect(result.stages);

        for (std::size_t i = 0; i < count; ++i) {
            result.verdicts[first + i] = verdict_of(status[i]);
            result.valid[first + i] = status[i] == 0 ? 1 : 0;
        }
    }

    result.elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return result;
}

} // namespace ppq
