/// GPU microbenchmarks: every kernel of the staged verifier on its own, the standalone NTT and
/// SHAKE kernels, and the GpuVerifier end to end.
///
/// Kernel benchmarks time only device work, between HIP events on a dedicated stream
/// (manual time); any input a kernel overwrites in place is restored before the start event.
/// SHAKE and the verifier are timed on the wall clock, since they include their copies.
///
///   bench_gpu.exe [--ppq_size=smoke|quick|full] [Google Benchmark flags]

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "bench_util.h"
#include "ppq/buffer.h"
#include "ppq/device.h"
#include "ppq/device_batch.h"
#include "ppq/gpu_verifier.h"
#include "ppq/keccak.h"
#include "ppq/ntt_kernels.h"
#include "ppq/shake_gpu.h"
#include "ppq/stream.h"
#include "ppq/verify_kernels.h"

namespace {

using bench::Size;
using benchmark::State;

constexpr int SKIP_EXIT_CODE = 77;

/// The geometry the pipeline is first run with to produce each stage's inputs
constexpr unsigned SETUP_THREADS = 128;
constexpr unsigned SETUP_NTT_GROUP = 32;

/// Runs the benchmark loop over device work: `before` is untimed setup on the stream (e.g.
/// restoring an input), `launch` is what is timed. Launcher errors (an unsupported block
/// size on this device, say) skip the benchmark with the message.
template <class Before, class Launch>
void time_device(State &state, const ppq::Stream &stream, Before before, Launch launch) {
    ppq::Event start;
    ppq::Event stop;
    try {
        for (auto _ : state) {
            before();
            start.record(stream.get());
            launch();
            stop.record(stream.get());
            stop.synchronize();
            state.SetIterationTime(ppq::elapsed_ms(start, stop) / 1e3);
        }
    } catch (const std::exception &e) {
        state.SkipWithError(e.what());
    }
}

template <class Launch> void time_device(State &state, const ppq::Stream &stream, Launch launch) {
    time_device(state, stream, [] {}, launch);
}

template <class T>
void copy_async(T *dst, const T *src, std::size_t count, const ppq::Stream &stream) {
    HIP_CHECK(hipMemcpyAsync(dst, src, count * sizeof(T), hipMemcpyDeviceToDevice, stream.get()));
}

std::string geometry_tag(unsigned threads) { return "/threads:" + std::to_string(threads); }

// ---------------------------------------------------------------------------------- SHAKE

void register_shake(Size size) {
    const auto inputs =
        bench::pick<std::size_t>(size, {32}, {32, 1312, 2592}, {32, 1312, 2592, 4627});
    const auto counts = bench::pick<std::size_t>(size, {256}, {16384}, {1024, 16384, 131072});
    struct Xof {
        const char *name;
        ppq::ShakeVariant variant;
        std::size_t out_len;
    };
    // SHAKE256 to 64 bytes (tr, mu), SHAKE128 to five blocks (about one ExpandA entry)
    const Xof xofs[] = {{"256", ppq::ShakeVariant::Shake256, 64},
                        {"128", ppq::ShakeVariant::Shake128, 5 * ppq::SHAKE128_RATE}};
    for (const Xof &x : xofs) {
        for (const std::size_t in_len : inputs) {
            for (const std::size_t count : counts) {
                const std::string name = std::string("shake_gpu/xof:") + x.name +
                                         "/in_bytes:" + std::to_string(in_len) +
                                         "/requests:" + std::to_string(count);
                bench::add(name,
                           [=](State &state) {
                               std::vector<std::uint8_t> data(in_len * count);
                               std::mt19937 rng(9);
                               for (std::uint8_t &b : data) {
                                   b = static_cast<std::uint8_t>(rng());
                               }
                               std::vector<ppq::ShakeRequest> reqs(count);
                               for (std::size_t i = 0; i < count; ++i) {
                                   reqs[i] = {x.variant, data.data() + i * in_len, in_len,
                                              x.out_len};
                               }
                               const ppq::Launch launch{(count + 127) / 128, 128};
                               for (auto _ : state) {
                                   auto out = ppq::shake_gpu(reqs, launch);
                                   benchmark::DoNotOptimize(out.data());
                               }
                               state.SetLabel("wall time, includes copies");
                               bench::set_items(state, count, count * in_len);
                           })
                    ->UseRealTime()
                    ->Unit(benchmark::kMicrosecond);
            }
        }
    }
}

// ------------------------------------------------------------------------------------- NTT

/// count random canonical polynomials on the device, with a pristine copy to restore from
struct NttData {
    ppq::DeviceBuffer<std::int32_t> polys;
    ppq::DeviceBuffer<std::int32_t> pristine;

    explicit NttData(std::size_t count) {
        std::vector<std::int32_t> host(count * ppq::N);
        std::mt19937 rng(17);
        std::uniform_int_distribution<std::int32_t> coeff(0, ppq::Q - 1);
        for (std::int32_t &c : host) {
            c = coeff(rng);
        }
        polys = ppq::DeviceBuffer<std::int32_t>(host);
        pristine = ppq::DeviceBuffer<std::int32_t>(host);
    }

    void restore(const ppq::Stream &s) { copy_async(polys.get(), pristine.get(), polys.size(), s); }
};

const char *dir_name(ppq::NttDirection d) {
    return d == ppq::NttDirection::Forward ? "forward" : "inverse";
}

/// The cooperative NTT must match the one-thread-per-polynomial baseline before it is timed
bool coop_matches_simple(std::size_t count, ppq::NttDirection dir, unsigned group,
                         unsigned threads) {
    NttData a(count);
    NttData b(count);
    ppq::launch_ntt_simple(a.polys.get(), count, dir, {(count + 63) / 64, 64}, nullptr);
    const std::size_t per_block = threads / group;
    ppq::launch_ntt_coop(
        b.polys.get(), count, dir, group,
        {std::min<std::size_t>((count + per_block - 1) / per_block, 65535), threads}, nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::int32_t> ha(count * ppq::N);
    std::vector<std::int32_t> hb(count * ppq::N);
    a.polys.download(ha);
    b.polys.download(hb);
    return ha == hb;
}

void register_ntt(Size size) {
    const auto counts = bench::pick<std::size_t>(size, {64}, {65536}, {4096, 65536, 524288});
    const auto block_sizes = bench::pick<unsigned>(size, {64}, {128, 256}, {64, 128, 256, 512});
    for (const ppq::NttDirection dir : {ppq::NttDirection::Forward, ppq::NttDirection::Inverse}) {
        for (const std::size_t count : counts) {
            const std::string tail = "/polys:" + std::to_string(count);
            for (const unsigned threads : block_sizes) {
                bench::add(
                    std::string("ntt_simple/dir:") + dir_name(dir) + geometry_tag(threads) + tail,
                    [=](State &state) {
                        ppq::Stream stream;
                        NttData d(count);
                        const ppq::Launch l{(count + threads - 1) / threads, threads};
                        time_device(
                            state, stream, [&] { d.restore(stream); },
                            [&] {
                                ppq::launch_ntt_simple(d.polys.get(), count, dir, l, stream.get());
                            });
                        bench::set_items(state, count);
                    })
                    ->UseManualTime()
                    ->Unit(benchmark::kMicrosecond);

                for (const unsigned group : ppq::NTT_GROUP_SIZES) {
                    if (threads % group != 0) {
                        continue;
                    }
                    bench::add(
                        std::string("ntt_coop/dir:") + dir_name(dir) +
                            "/group:" + std::to_string(group) + geometry_tag(threads) + tail,
                        [=](State &state) {
                            try {
                                if (!coop_matches_simple(std::min<std::size_t>(count, 256), dir,
                                                         group, threads)) {
                                    state.SkipWithError("differs from launch_ntt_simple");
                                    return;
                                }
                            } catch (const std::exception &e) {
                                state.SkipWithError(e.what());
                                return;
                            }
                            ppq::Stream stream;
                            NttData d(count);
                            const std::size_t per_block = threads / group;
                            const ppq::Launch l{
                                std::min<std::size_t>((count + per_block - 1) / per_block, 65535),
                                threads};
                            time_device(
                                state, stream, [&] { d.restore(stream); },
                                [&] {
                                    ppq::launch_ntt_coop(d.polys.get(), count, dir, group, l,
                                                         stream.get());
                                });
                            bench::set_items(state, count);
                        })
                        ->UseManualTime()
                        ->Unit(benchmark::kMicrosecond);
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------ key stages

struct KeyTable {
    ppq::DeviceKeys keys;
    ppq::KeyCacheBuffers cache;

    KeyTable(const ppq::Workload &w) : keys(w.pks, w.key_count()), cache(w.params, w.key_count()) {}
};

void register_key_stages(Size size) {
    const auto key_counts = bench::pick<std::size_t>(size, {16}, {1024}, {64, 1024, 4096});
    const auto block_sizes = bench::pick<unsigned>(size, {64}, {128}, {64, 128, 256});
    using KeyLauncher = void (*)(const ppq::ParamSet &, const ppq::KeyInputs &,
                                 const ppq::KeyCacheView &, unsigned, hipStream_t);
    struct Stage {
        const char *name;
        KeyLauncher launch;
    };
    const Stage stages[] = {{"key_tr", &ppq::launch_key_tr},
                            {"key_expand_a", &ppq::launch_key_expand_a},
                            {"key_t1", &ppq::launch_key_t1},
                            {"prepare_keys", &ppq::launch_prepare_keys}};
    for (const ppq::ParamSet *p : bench::param_sets(size)) {
        for (const std::size_t keys : key_counts) {
            for (const Stage &st : stages) {
                for (const unsigned threads : block_sizes) {
                    const std::string name =
                        std::string("gpu_stage/") + st.name + "/params:" + bench::params_label(*p) +
                        "/keys:" + std::to_string(keys) + geometry_tag(threads);
                    bench::add(name,
                               [=](State &state) {
                                   const ppq::Workload &w = bench::workload(*p, keys, keys);
                                   KeyTable t(w);
                                   ppq::Stream stream;
                                   const ppq::KeyCacheView view = t.cache.view(keys);
                                   time_device(state, stream, [&] {
                                       st.launch(*p, t.keys.view(), view, threads, stream.get());
                                   });
                                   bench::set_items(state, keys);
                               })
                        ->UseManualTime()
                        ->Unit(benchmark::kMicrosecond);
                }
            }
        }
    }
}

// ------------------------------------------------------------------------ signature stages

/// One chunk run through every stage once, so each stage's inputs are in place, with copies
/// of the two buffers the NTT stages transform in place. Throws if the run does not verify
/// every signature: a benchmark of a broken pipeline would measure nothing useful.
class Pipeline {
  public:
    Pipeline(const ppq::Workload &w)
        : p(w.params), sigs(w.sig_count()), keys_in(w.pks, w.key_count()),
          cache(w.params, w.key_count()), batch(w, 0, w.sig_count()), work(w.params, sigs),
          key_view(cache.view(w.key_count())), in(batch.view()), sw(work.view()),
          z_decoded(sigs * work.strides().z), w_product(sigs * work.strides().w) {
        const unsigned t = SETUP_THREADS;
        const unsigned g = SETUP_NTT_GROUP;
        const hipStream_t s = stream.get();
        ppq::launch_prepare_keys(p, keys_in.view(), key_view, t, s);
        ppq::launch_decode(p, in, sw, t, s);
        copy_async(z_decoded.get(), sw.z, z_decoded.size(), stream);
        ppq::launch_mu(p, in, key_view, sw, t, s);
        ppq::launch_challenge(p, in, sw, t, s);
        ppq::launch_z_ntt(p, in, sw, g, t, s);
        ppq::launch_matvec(p, in, key_view, sw, t, s);
        copy_async(w_product.get(), sw.w, w_product.size(), stream);
        ppq::launch_w_intt(p, in, sw, g, t, s);
        ppq::launch_use_hint(p, in, sw, t, s);
        ppq::launch_final_hash(p, in, sw, t, s);
        stream.synchronize();
        const auto status = work.status(sigs);
        if (std::any_of(status.begin(), status.end(), [](std::uint32_t v) { return v != 0; })) {
            throw std::runtime_error("the setup run did not verify every signature");
        }
    }

    void restore_z() { copy_async(sw.z, z_decoded.get(), z_decoded.size(), stream); }
    void restore_w() { copy_async(sw.w, w_product.get(), w_product.size(), stream); }

    const ppq::ParamSet p;
    const std::size_t sigs;
    ppq::Stream stream;
    ppq::DeviceKeys keys_in;
    ppq::KeyCacheBuffers cache;
    ppq::DeviceSigBatch batch;
    ppq::SigWorkBuffers work;
    const ppq::KeyCacheView key_view;
    const ppq::SigInputs in;
    const ppq::SigWork sw;

  private:
    ppq::DeviceBuffer<std::int32_t> z_decoded;
    ppq::DeviceBuffer<std::int32_t> w_product;
};

struct PipelineCache {
    int params = 0;
    std::size_t sigs = 0;
    std::size_t keys = 0;
    std::unique_ptr<Pipeline> pipe;
};

PipelineCache &pipeline_cache() {
    static PipelineCache cached;
    return cached;
}

/// The pipeline for the most recent (params, sigs, keys): registration groups benchmarks by
/// it, and one at a time keeps device memory bounded
Pipeline &pipeline(const ppq::ParamSet &p, std::size_t sigs, std::size_t keys) {
    PipelineCache &c = pipeline_cache();
    if (!c.pipe || c.params != p.id || c.sigs != sigs || c.keys != keys) {
        c = PipelineCache{}; // free the previous one first
        c.pipe = std::make_unique<Pipeline>(bench::workload(p, sigs, keys));
        c.params = p.id;
        c.sigs = sigs;
        c.keys = keys;
    }
    return *c.pipe;
}

enum class StageKind {
    Plain,
    /// Has an NTT group size argument
    Ntt,
};

struct SigStage {
    const char *name;
    StageKind kind;
    /// Depends on which key each signature uses (reads the key cache)
    bool reads_keys;
    void (*run)(Pipeline &, unsigned threads, unsigned group);
    void (*before)(Pipeline &);
};

void no_setup(Pipeline &) {}

const SigStage SIG_STAGES[] = {
    {"decode", StageKind::Plain, false,
     [](Pipeline &x, unsigned t, unsigned) {
         ppq::launch_decode(x.p, x.in, x.sw, t, x.stream.get());
     },
     &no_setup},
    {"mu", StageKind::Plain, true,
     [](Pipeline &x, unsigned t, unsigned) {
         ppq::launch_mu(x.p, x.in, x.key_view, x.sw, t, x.stream.get());
     },
     &no_setup},
    {"challenge", StageKind::Plain, false,
     [](Pipeline &x, unsigned t, unsigned) {
         ppq::launch_challenge(x.p, x.in, x.sw, t, x.stream.get());
     },
     &no_setup},
    {"ntt_z", StageKind::Ntt, false,
     [](Pipeline &x, unsigned t, unsigned g) {
         ppq::launch_z_ntt(x.p, x.in, x.sw, g, t, x.stream.get());
     },
     [](Pipeline &x) { x.restore_z(); }},
    {"matvec", StageKind::Plain, true,
     [](Pipeline &x, unsigned t, unsigned) {
         ppq::launch_matvec(x.p, x.in, x.key_view, x.sw, t, x.stream.get());
     },
     &no_setup},
    {"intt_w", StageKind::Ntt, false,
     [](Pipeline &x, unsigned t, unsigned g) {
         ppq::launch_w_intt(x.p, x.in, x.sw, g, t, x.stream.get());
     },
     [](Pipeline &x) { x.restore_w(); }},
    {"use_hint", StageKind::Plain, false,
     [](Pipeline &x, unsigned t, unsigned) {
         ppq::launch_use_hint(x.p, x.in, x.sw, t, x.stream.get());
     },
     &no_setup},
    {"final_hash", StageKind::Plain, false,
     [](Pipeline &x, unsigned t, unsigned) {
         ppq::launch_final_hash(x.p, x.in, x.sw, t, x.stream.get());
     },
     &no_setup},
};

void register_sig_stages(Size size) {
    const auto sig_counts = bench::pick<std::size_t>(size, {64}, {8192}, {1024, 8192, 65536});
    // One key shared by every signature, or many (the key cache no longer fits in cache)
    const auto key_counts = bench::pick<std::size_t>(size, {1}, {1, 1024}, {1, 1024});
    const auto block_sizes = bench::pick<unsigned>(size, {64}, {128}, {64, 128, 256, 512});
    const auto groups = bench::pick<unsigned>(size, {32}, {16, 32, 64, 128}, {16, 32, 64, 128});

    for (const ppq::ParamSet *p : bench::param_sets(size)) {
        for (const std::size_t sigs : sig_counts) {
            for (std::size_t ki = 0; ki < key_counts.size(); ++ki) {
                const std::size_t keys = std::min(key_counts[ki], sigs);
                if (ki > 0 && keys == key_counts[0]) {
                    continue;
                }
                for (const SigStage &st : SIG_STAGES) {
                    // Stages that never read the key cache are timed once per batch size
                    if (ki > 0 && !st.reads_keys) {
                        continue;
                    }
                    const std::vector<unsigned> stage_groups =
                        st.kind == StageKind::Ntt ? groups : std::vector<unsigned>{0};
                    for (const unsigned group : stage_groups) {
                        for (const unsigned threads : block_sizes) {
                            if (group != 0 && threads % group != 0) {
                                continue;
                            }
                            std::string name = std::string("gpu_stage/") + st.name +
                                               "/params:" + bench::params_label(*p) +
                                               "/sigs:" + std::to_string(sigs);
                            if (st.reads_keys) {
                                name += "/keys:" + std::to_string(keys);
                            }
                            if (group != 0) {
                                name += "/group:" + std::to_string(group);
                            }
                            name += geometry_tag(threads);
                            bench::add(name,
                                       [=](State &state) {
                                           Pipeline *x = nullptr;
                                           try {
                                               x = &pipeline(*p, sigs, keys);
                                           } catch (const std::exception &e) {
                                               state.SkipWithError(e.what());
                                               return;
                                           }
                                           time_device(
                                               state, x->stream, [&] { st.before(*x); },
                                               [&] { st.run(*x, threads, group); });
                                           bench::set_items(state, sigs);
                                       })
                                ->UseManualTime()
                                ->Unit(benchmark::kMicrosecond);
                        }
                    }
                }
            }
        }
    }
}

// --------------------------------------------------------------------- GpuVerifier end to end

void register_verifier(Size size) {
    const auto sig_counts = bench::pick<std::size_t>(size, {64}, {16384}, {1024, 16384, 65536});
    const auto chunks = bench::pick<std::size_t>(size, {32}, {8192}, {2048, 8192, 32768});
    for (const ppq::ParamSet *p : bench::param_sets(size)) {
        for (const std::size_t sigs : sig_counts) {
            // All signatures under one key (one shared key table), or each under its own (the
            // keys are prepared per chunk)
            for (const std::size_t keys : {std::size_t{1}, sigs}) {
                for (const std::size_t chunk : chunks) {
                    ppq::GpuVerifierOptions o;
                    o.chunk = chunk;
                    const std::string name = "gpu_verifier/params:" + bench::params_label(*p) +
                                             "/sigs:" + std::to_string(sigs) +
                                             "/keys:" + std::to_string(keys) +
                                             "/chunk:" + std::to_string(chunk);
                    bench::add(name,
                               [=](State &state) {
                                   pipeline_cache() = PipelineCache{}; // its device memory
                                   const ppq::Workload &w = bench::workload(*p, sigs, keys);
                                   try {
                                       ppq::GpuVerifier v(o);
                                       if (!ppq::compare(w, v.verify(w)).ok()) {
                                           state.SkipWithError(
                                               "results disagree with the workload");
                                           return;
                                       }
                                       std::vector<ppq::StageTime> totals;
                                       for (auto _ : state) {
                                           const ppq::VerifyResult r = v.verify(w);
                                           for (const ppq::StageTime &t : r.stages) {
                                               auto it = std::find_if(totals.begin(), totals.end(),
                                                                      [&](const ppq::StageTime &x) {
                                                                          return x.name == t.name;
                                                                      });
                                               if (it == totals.end()) {
                                                   totals.push_back(t);
                                               } else {
                                                   it->ms += t.ms;
                                               }
                                           }
                                       }
                                       // Mean milliseconds per verify() in each stage
                                       for (const ppq::StageTime &t : totals) {
                                           state.counters[t.name + "_ms"] = benchmark::Counter(
                                               t.ms, benchmark::Counter::kAvgIterations);
                                       }
                                   } catch (const std::exception &e) {
                                       state.SkipWithError(e.what());
                                       return;
                                   }
                                   bench::set_items(state, sigs);
                               })
                        ->UseRealTime()
                        ->Unit(benchmark::kMillisecond);
                }
            }
        }
    }
}

} // namespace

int main(int argc, char **argv) {
    const Size size = bench::take_size_flag(argc, argv);
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 2;
    }

    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
        std::cout << "no HIP device visible; skipping GPU benchmarks\n";
        return SKIP_EXIT_CODE;
    }
    const ppq::DeviceInfo dev = ppq::open_device(0);
    benchmark::AddCustomContext("gpu", dev.name + " (" + dev.arch + ", " +
                                           std::to_string(dev.compute_units) + " CUs)");
    benchmark::AddCustomContext("gpu_platform", ppq::platform_name());
    bench::add_common_context(size);

    register_shake(size);
    register_ntt(size);
    register_key_stages(size);
    register_sig_stages(size);
    register_verifier(size);

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
