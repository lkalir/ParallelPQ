/// The decode stage on the GPU: z, h and the status bits for generated workloads (every tamper
/// mode) and the ACVP vectors, against the naive decoder

#include <hip/hip_runtime.h>

#include "gpu_test.h"
#include "naive_mldsa.h"
#include "ppq/acvp.h"
#include "ppq/device_batch.h"
#include "ppq/generate.h"
#include "ppq/verify_kernels.h"

namespace {

using naive::Bytes;

ppq::Workload mixed(const ppq::ParamSet &p, std::size_t n, std::uint64_t seed) {
    ppq::GenSpec s;
    s.params = p;
    s.signatures = n;
    s.keys = 3;
    s.pass_rate = 0.5;
    s.fail_modes = {ppq::Tamper::Message, ppq::Tamper::Commitment, ppq::Tamper::ZNorm,
                    ppq::Tamper::Hint};
    s.msg_min = 0;
    s.msg_max = 40;
    s.ctx_max = 8;
    s.seed = seed;
    s.threads = 2;
    return ppq::generate(s);
}

/// Decodes signatures [first, first + count) on the GPU and checks every output against the
/// naive decoder; returns the number of signatures with any mismatch
std::size_t check_decode(const ppq::Workload &w, std::size_t first, std::size_t count,
                         unsigned threads) {
    const ppq::ParamSet &p = w.params;
    const ppq::DeviceSigBatch batch(w, first, count);
    ppq::SigWorkBuffers work(p, count);
    ppq::launch_decode(p, batch.view(), work.view(), threads, nullptr);
    HIP_CHECK(hipDeviceSynchronize());

    const auto status = work.status(count);
    const auto z = work.z(count);
    const auto h = work.h(count);
    const auto zs = work.strides().z;
    const auto hs = work.strides().h;

    std::size_t wrong = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const Bytes sig(w.sig(first + i).begin(), w.sig(first + i).end());
        const auto nz = naive::z_decode(p, sig);
        const auto nh = naive::hint_decode(p, sig);
        const bool z_bad = naive::max_abs(nz) >= p.gamma1 - p.beta;

        bool ok = (status[i] ==
                   ((nh ? 0u : ppq::STATUS_HINT_BAD) | (z_bad ? ppq::STATUS_ZNORM_BAD : 0u)));
        for (std::size_t j = 0; j < nz.size(); ++j) {
            ok = ok && std::equal(nz[j].begin(), nz[j].end(),
                                  z.begin() + static_cast<std::ptrdiff_t>(i * zs + j * ppq::N));
        }
        if (nh) {
            for (std::size_t k = 0; k < nh->size(); ++k) {
                for (unsigned c = 0; c < ppq::N; ++c) {
                    ok = ok && h[i * hs + k * ppq::N + c] == static_cast<std::uint8_t>((*nh)[k][c]);
                }
            }
        }
        // The decode stage alone decides the bad-hint and z-norm verdicts
        const ppq::Verdict e = w.expected[first + i];
        if (e == ppq::Verdict::BadHint || e == ppq::Verdict::ZNorm) {
            ok = ok && ppq::verdict_of(status[i]) == e;
        } else {
            ok = ok && status[i] == 0;
        }
        wrong += !ok;
    }
    return wrong;
}

TEST_CASE(generated_workloads_decode) {
    for (const ppq::ParamSet &p : ppq::PARAM_SETS) {
        const ppq::Workload w = mixed(p, 24, 300 + static_cast<std::uint64_t>(p.id));
        for (const unsigned threads : {1u, 32u, 128u}) {
            CHECK(check_decode(w, 0, w.sig_count(), threads) == 0);
        }
    }
}

TEST_CASE(acvp_vectors_decode) {
    for (const ppq::AcvpWorkload &g :
         ppq::load_acvp_sigver(TEST_VECTORS_DIR "/acvp/mldsa_sigver.txt")) {
        CHECK(check_decode(g.workload, 0, g.workload.sig_count(), 64) == 0);
    }
}

TEST_CASE(chunks_decode_their_own_signatures) {
    const ppq::Workload w = mixed(ppq::ML_DSA_65, 20, 400);
    CHECK(check_decode(w, 5, 7, 32) == 0);
    CHECK(check_decode(w, 19, 1, 32) == 0);
    CHECK(check_decode(w, 0, 0, 32) == 0);
}

TEST_CASE(status_is_reset_between_batches) {
    const ppq::ParamSet &p = ppq::ML_DSA_44;
    ppq::GenSpec bad;
    bad.params = p;
    bad.signatures = 8;
    bad.pass_rate = 0.0;
    bad.fail_modes = {ppq::Tamper::Hint, ppq::Tamper::ZNorm};
    bad.threads = 2;
    ppq::GenSpec good = bad;
    good.pass_rate = 1.0;

    ppq::SigWorkBuffers work(p, 8);
    const ppq::DeviceSigBatch b1(ppq::generate(bad), 0, 8);
    ppq::launch_decode(p, b1.view(), work.view(), 32, nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    const auto s1 = work.status(8);
    CHECK(std::all_of(s1.begin(), s1.end(), [](std::uint32_t s) { return s != 0; }));

    const ppq::DeviceSigBatch b2(ppq::generate(good), 0, 8);
    ppq::launch_decode(p, b2.view(), work.view(), 32, nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    const auto s2 = work.status(8);
    CHECK(std::all_of(s2.begin(), s2.end(), [](std::uint32_t s) { return s == 0; }));
}

} // namespace

int main() { return gpu_test::run_all(); }
