/// ExpandA and the challenge on the GPU, against the stage vectors and the naive transcription

#include <hip/hip_runtime.h>

#include <map>

#include "gpu_test.h"
#include "naive_mldsa.h"
#include "ppq/acvp.h"
#include "ppq/device_batch.h"
#include "ppq/generate.h"
#include "ppq/reduce.h"
#include "ppq/verify_kernels.h"
#include "vector_util.h"

namespace {

using naive::Bytes;
using naive::Poly;

/// Runs ExpandA for these keys; returns the number of wrong polynomials
std::size_t check_expand_a(const ppq::ParamSet &p, const std::vector<Bytes> &pks,
                           const std::vector<std::vector<Poly>> *expected, unsigned threads) {
    std::vector<std::uint8_t> flat;
    for (const Bytes &pk : pks) {
        flat.insert(flat.end(), pk.begin(), pk.end());
    }
    const ppq::DeviceKeys keys(flat, pks.size());
    ppq::KeyCacheBuffers cache(p, pks.size());
    ppq::launch_key_expand_a(p, keys.view(), cache.view(pks.size()), threads, nullptr);
    HIP_CHECK(hipDeviceSynchronize());

    const std::size_t stride = cache.strides().a_hat;
    std::vector<std::int32_t> a(pks.size() * stride);
    HIP_CHECK(hipMemcpy(a.data(), cache.view(pks.size()).a_hat, a.size() * sizeof(std::int32_t),
                        hipMemcpyDeviceToHost));

    std::size_t wrong = 0;
    for (std::size_t k = 0; k < pks.size(); ++k) {
        const auto want = expected != nullptr
                              ? (*expected)[k]
                              : naive::expand_a(p, Bytes(pks[k].begin(), pks[k].begin() + 32));
        for (std::size_t e = 0; e < want.size(); ++e) {
            wrong += !std::equal(want[e].begin(), want[e].end(),
                                 a.begin() + static_cast<std::ptrdiff_t>(k * stride + e * ppq::N));
        }
    }
    return wrong;
}

TEST_CASE(expand_a_matches_the_stage_vectors) {
    std::map<int, std::vector<Bytes>> pks;
    std::map<int, std::vector<std::vector<Poly>>> want;
    for (const ppq::KvRecord &r :
         ppq::load_kv_records(TEST_VECTORS_DIR "/stages/mldsa_stages.txt")) {
        if (r.str("kind") == "key") {
            const ppq::ParamSet &p = vector_util::params_of(r);
            pks[p.id].push_back(r.bytes("pk"));
            want[p.id].push_back(
                vector_util::polys(r, "a_hat", static_cast<std::size_t>(p.k * p.l)));
        }
    }
    for (const ppq::ParamSet &p : ppq::PARAM_SETS) {
        CHECK(check_expand_a(p, pks[p.id], &want[p.id], 64) == 0);
    }
}

TEST_CASE(expand_a_matches_naive_for_generated_keys) {
    for (const ppq::ParamSet &p : ppq::PARAM_SETS) {
        ppq::GenSpec s;
        s.params = p;
        s.signatures = 3;
        s.seed = 700;
        s.threads = 2;
        const ppq::Workload w = ppq::generate(s);
        std::vector<Bytes> pks;
        for (std::size_t k = 0; k < w.key_count(); ++k) {
            pks.emplace_back(w.pk(k).begin(), w.pk(k).end());
        }
        for (const unsigned threads : {1u, 32u, 128u}) {
            CHECK(check_expand_a(p, pks, nullptr, threads) == 0);
        }
    }
}

std::size_t check_challenge(const ppq::Workload &w, unsigned threads) {
    const ppq::ParamSet &p = w.params;
    const ppq::DeviceSigBatch batch(w, 0, w.sig_count());
    ppq::SigWorkBuffers work(p, w.sig_count());
    ppq::launch_challenge(p, batch.view(), work.view(), threads, nullptr);
    HIP_CHECK(hipDeviceSynchronize());

    const auto c = work.c(w.sig_count());
    std::size_t wrong = 0;
    for (std::size_t i = 0; i < w.sig_count(); ++i) {
        const Bytes ctilde(w.sig(i).begin(),
                           w.sig(i).begin() + static_cast<std::ptrdiff_t>(p.ctilde_bytes));
        const Poly want = naive::ntt(naive::sample_in_ball(p, ctilde));
        for (unsigned j = 0; j < ppq::N; ++j) {
            if (ppq::freeze(c[i * ppq::N + j]) != want[j]) {
                ++wrong;
                break;
            }
        }
    }
    return wrong;
}

TEST_CASE(challenge_matches_naive) {
    for (const ppq::ParamSet &p : ppq::PARAM_SETS) {
        ppq::GenSpec s;
        s.params = p;
        s.signatures = 20;
        s.keys = 2;
        s.pass_rate = 0.5;
        s.fail_modes = {ppq::Tamper::Commitment};
        s.seed = 800;
        s.threads = 2;
        const ppq::Workload w = ppq::generate(s);
        CHECK(check_challenge(w, 32) == 0);
        CHECK(check_challenge(w, 1) == 0);
    }
    for (const ppq::AcvpWorkload &g :
         ppq::load_acvp_sigver(TEST_VECTORS_DIR "/acvp/mldsa_sigver.txt")) {
        CHECK(check_challenge(g.workload, 64) == 0);
    }
}

} // namespace

int main() { return gpu_test::run_all(); }
