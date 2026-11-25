/// The key cache on the GPU (tr, A_hat, NTT(t1 * 2^d)) against the stage vectors and the
/// pq-crystals stage internals

#include <hip/hip_runtime.h>

#include <map>

#include "gpu_test.h"
#include "ppq/acvp.h"
#include "ppq/cpuref_stages.h"
#include "ppq/device_batch.h"
#include "ppq/generate.h"
#include "ppq/reduce.h"
#include "ppq/verify_kernels.h"
#include "vector_util.h"

namespace {

using ppq::cpuref::Bytes;
using ppq::cpuref::KeyStages;

struct Cache {
    std::vector<std::uint8_t> tr;
    std::vector<std::int32_t> a_hat;
    std::vector<std::int32_t> t1_hat;
};

Cache prepare(const ppq::ParamSet &p, const std::vector<std::uint8_t> &pks, std::size_t count,
              unsigned threads) {
    const ppq::DeviceKeys keys(pks, count);
    ppq::KeyCacheBuffers cache(p, count);
    const ppq::KeyCacheView v = cache.view(count);
    ppq::launch_prepare_keys(p, keys.view(), v, threads, nullptr);
    HIP_CHECK(hipDeviceSynchronize());

    const ppq::KeyStrides s = cache.strides();
    Cache out{std::vector<std::uint8_t>(count * s.tr), std::vector<std::int32_t>(count * s.a_hat),
              std::vector<std::int32_t>(count * s.t1_hat)};
    if (count > 0) {
        HIP_CHECK(hipMemcpy(out.tr.data(), v.tr, out.tr.size(), hipMemcpyDeviceToHost));
        HIP_CHECK(
            hipMemcpy(out.a_hat.data(), v.a_hat, out.a_hat.size() * 4, hipMemcpyDeviceToHost));
        HIP_CHECK(
            hipMemcpy(out.t1_hat.data(), v.t1_hat, out.t1_hat.size() * 4, hipMemcpyDeviceToHost));
    }
    return out;
}

/// Does key k of the GPU cache equal these (canonical) stages?
bool matches(const ppq::ParamSet &p, const Cache &c, std::size_t k, const KeyStages &want) {
    const ppq::KeyStrides s = ppq::key_strides(p);
    bool ok = std::equal(want.tr.begin(), want.tr.end(),
                         c.tr.begin() + static_cast<std::ptrdiff_t>(k * s.tr));
    for (std::size_t e = 0; e < want.a_hat.size(); ++e) {
        for (unsigned j = 0; j < ppq::N; ++j) {
            ok = ok && c.a_hat[k * s.a_hat + e * ppq::N + j] == want.a_hat[e][j];
        }
    }
    for (std::size_t i = 0; i < want.t1_hat.size(); ++i) {
        for (unsigned j = 0; j < ppq::N; ++j) {
            ok = ok && ppq::freeze(c.t1_hat[k * s.t1_hat + i * ppq::N + j]) == want.t1_hat[i][j];
        }
    }
    return ok;
}

TEST_CASE(stage_vector_keys) {
    std::map<int, std::vector<std::uint8_t>> pks;
    std::map<int, std::vector<KeyStages>> want;
    for (const ppq::KvRecord &r :
         ppq::load_kv_records(TEST_VECTORS_DIR "/stages/mldsa_stages.txt")) {
        if (r.str("kind") != "key") {
            continue;
        }
        const ppq::ParamSet &p = vector_util::params_of(r);
        const Bytes pk = r.bytes("pk");
        pks[p.id].insert(pks[p.id].end(), pk.begin(), pk.end());
        KeyStages ks;
        ks.tr = r.bytes("tr");
        ks.a_hat = vector_util::polys(r, "a_hat", static_cast<std::size_t>(p.k * p.l));
        ks.t1_hat = vector_util::polys(r, "t1_hat", static_cast<std::size_t>(p.k));
        want[p.id].push_back(ks);
    }
    for (const ppq::ParamSet &p : ppq::PARAM_SETS) {
        const Cache c = prepare(p, pks[p.id], want[p.id].size(), 64);
        for (std::size_t k = 0; k < want[p.id].size(); ++k) {
            CHECK(matches(p, c, k, want[p.id][k]));
        }
    }
}

TEST_CASE(generated_and_acvp_keys_match_pq_crystals) {
    std::vector<ppq::Workload> workloads;
    for (const ppq::ParamSet &p : ppq::PARAM_SETS) {
        ppq::GenSpec s;
        s.params = p;
        s.signatures = 5;
        s.seed = 900;
        s.threads = 2;
        workloads.push_back(ppq::generate(s));
    }
    for (ppq::AcvpWorkload &g : ppq::load_acvp_sigver(TEST_VECTORS_DIR "/acvp/mldsa_sigver.txt")) {
        workloads.push_back(std::move(g.workload));
    }

    for (const ppq::Workload &w : workloads) {
        for (const unsigned threads : {1u, 32u}) {
            const Cache c = prepare(w.params, w.pks, w.key_count(), threads);
            for (std::size_t k = 0; k < w.key_count(); ++k) {
                CHECK(matches(w.params, c, k, ppq::cpuref::key_stages(w.params, w.pk(k))));
            }
        }
    }
}

TEST_CASE(cache_must_hold_the_key_table) {
    const ppq::ParamSet &p = ppq::ML_DSA_44;
    const ppq::DeviceKeys keys(std::vector<std::uint8_t>(2 * p.pk_bytes), 2);
    ppq::KeyCacheBuffers cache(p, 1);
    bool threw = false;
    try {
        ppq::KeyCacheView v = cache.view(1);
        ppq::launch_prepare_keys(p, keys.view(), v, 32, nullptr);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
    // An empty key table is a no-op
    const Cache none = prepare(p, {}, 0, 32);
    CHECK(none.tr.empty());
}

} // namespace

int main() { return gpu_test::run_all(); }
