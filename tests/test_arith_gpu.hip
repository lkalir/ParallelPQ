/// The per-signature arithmetic on the GPU (NTT(z), the matrix-vector product, NTT^-1, UseHint
/// and w1Encode), checked stage by stage against the pq-crystals stage internals

#include <hip/hip_runtime.h>

#include "gpu_test.h"
#include "ppq/acvp.h"
#include "ppq/cpuref_stages.h"
#include "ppq/device_batch.h"
#include "ppq/generate.h"
#include "ppq/reduce.h"
#include "ppq/verify_kernels.h"

namespace {

struct Geometry {
    unsigned threads;
    unsigned ntt_group;
};

ppq::cpuref::Bytes mu_for(const ppq::Workload &w, std::size_t i) {
    switch (w.iface) {
    case ppq::Interface::External:
        return ppq::cpuref::mu_external(w.params, w.pk_of(i), w.msg(i), w.ctx(i));
    case ppq::Interface::Internal:
        return ppq::cpuref::mu_internal(w.params, w.pk_of(i), w.msg(i));
    case ppq::Interface::Mu:
        break;
    }
    return {w.msg(i).begin(), w.msg(i).end()};
}

/// Runs every stage up to w1Encode and compares z_hat, w_approx and w1 with pq-crystals for
/// each signature whose hint decodes; returns the number of mismatching signatures
std::size_t check_arithmetic(const ppq::Workload &w, Geometry g) {
    const ppq::ParamSet &p = w.params;
    const std::size_t n = w.sig_count();
    const ppq::DeviceKeys keys(w.pks, w.key_count());
    ppq::KeyCacheBuffers cache(p, w.key_count());
    const ppq::DeviceSigBatch batch(w, 0, n);
    ppq::SigWorkBuffers work(p, n);
    const ppq::KeyCacheView kc = cache.view(w.key_count());
    const ppq::SigInputs in = batch.view();
    const ppq::SigWork sw = work.view();

    ppq::launch_prepare_keys(p, keys.view(), kc, g.threads, nullptr);
    ppq::launch_decode(p, in, sw, g.threads, nullptr);
    ppq::launch_challenge(p, in, sw, g.threads, nullptr);
    ppq::launch_z_ntt(p, in, sw, g.ntt_group, g.threads, nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    const auto z_hat = work.z(n);

    ppq::launch_matvec(p, in, kc, sw, g.threads, nullptr);
    ppq::launch_w_intt(p, in, sw, g.ntt_group, g.threads, nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    const auto w_approx = work.w(n);

    ppq::launch_use_hint(p, in, sw, g.threads, nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    const auto w1 = work.w1(n);

    const ppq::WorkStrides st = work.strides();
    std::size_t wrong = 0;
    std::size_t compared = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const auto want = ppq::cpuref::sig_stages(p, w.pk_of(i), mu_for(w, i), w.sig(i));
        bool ok = true;
        for (std::size_t j = 0; j < want.z_hat.size(); ++j) {
            for (unsigned c = 0; c < ppq::N; ++c) {
                ok = ok && ppq::freeze(z_hat[i * st.z + j * ppq::N + c]) == want.z_hat[j][c];
            }
        }
        if (want.hint_ok) {
            ++compared;
            for (std::size_t r = 0; r < want.w_approx.size(); ++r) {
                for (unsigned c = 0; c < ppq::N; ++c) {
                    ok = ok &&
                         ppq::freeze(w_approx[i * st.w + r * ppq::N + c]) == want.w_approx[r][c];
                }
            }
            ok = ok && std::equal(want.w1_packed.begin(), want.w1_packed.end(),
                                  w1.begin() + static_cast<std::ptrdiff_t>(i * st.w1));
        }
        wrong += !ok;
    }
    CHECK(compared > 0 || n == 0);
    return wrong;
}

ppq::Workload mixed(const ppq::ParamSet &p, std::size_t n, std::uint64_t seed) {
    ppq::GenSpec s;
    s.params = p;
    s.signatures = n;
    s.keys = 3;
    s.pass_rate = 0.5;
    s.fail_modes = {ppq::Tamper::Message, ppq::Tamper::Commitment, ppq::Tamper::ZNorm,
                    ppq::Tamper::Hint};
    s.msg_min = 0;
    s.msg_max = 64;
    s.ctx_max = 16;
    s.seed = seed;
    s.threads = 2;
    return ppq::generate(s);
}

TEST_CASE(generated_workloads_every_geometry) {
    for (const ppq::ParamSet &p : ppq::PARAM_SETS) {
        const ppq::Workload w = mixed(p, 16, 1000 + static_cast<std::uint64_t>(p.id));
        for (const Geometry g : {Geometry{128, 32}, Geometry{256, 128}, Geometry{64, 16}}) {
            CHECK(check_arithmetic(w, g) == 0);
        }
    }
}

TEST_CASE(acvp_every_interface) {
    for (const ppq::AcvpWorkload &g :
         ppq::load_acvp_sigver(TEST_VECTORS_DIR "/acvp/mldsa_sigver.txt")) {
        CHECK(check_arithmetic(g.workload, {128, 64}) == 0);
    }
}

TEST_CASE(threads_must_suit_the_ntt_group) {
    const ppq::Workload w = mixed(ppq::ML_DSA_44, 4, 1100);
    bool threw = false;
    try {
        check_arithmetic(w, {48, 32});
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}

} // namespace

int main() { return gpu_test::run_all(); }
