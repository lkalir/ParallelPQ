/// The tr and mu stages on the GPU: generated workloads with ragged messages and contexts, and
/// every ACVP interface, against the CPU reference and the naive transcription

#include <hip/hip_runtime.h>

#include "gpu_test.h"
#include "naive_mldsa.h"
#include "ppq/acvp.h"
#include "ppq/cpuref_stages.h"
#include "ppq/device_batch.h"
#include "ppq/generate.h"
#include "ppq/verify_kernels.h"

namespace {

using naive::Bytes;

Bytes expected_mu(const ppq::Workload &w, std::size_t i) {
    const Bytes pk(w.pk_of(i).begin(), w.pk_of(i).end());
    const Bytes msg(w.msg(i).begin(), w.msg(i).end());
    switch (w.iface) {
    case ppq::Interface::External:
        return naive::mu_external(pk, msg, Bytes(w.ctx(i).begin(), w.ctx(i).end()));
    case ppq::Interface::Internal:
        return ppq::cpuref::mu_internal(w.params, pk, msg);
    case ppq::Interface::Mu:
        return msg;
    }
    return {};
}

/// Runs the tr and mu stages for signatures [first, first + count); returns mismatches
std::size_t check_mu(const ppq::Workload &w, std::size_t first, std::size_t count,
                     unsigned threads) {
    const ppq::ParamSet &p = w.params;
    const ppq::DeviceKeys keys(w.pks, w.key_count());
    ppq::KeyCacheBuffers cache(p, w.key_count());
    const ppq::DeviceSigBatch batch(w, first, count);
    ppq::SigWorkBuffers work(p, count);

    ppq::launch_key_tr(p, keys.view(), cache.view(w.key_count()), threads, nullptr);
    ppq::launch_mu(p, batch.view(), cache.view(w.key_count()), work.view(), threads, nullptr);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<std::uint8_t> tr(w.key_count() * ppq::TR_BYTES);
    if (!tr.empty()) {
        HIP_CHECK(
            hipMemcpy(tr.data(), cache.view(w.key_count()).tr, tr.size(), hipMemcpyDeviceToHost));
    }
    std::size_t wrong = 0;
    for (std::size_t k = 0; k < w.key_count(); ++k) {
        const Bytes want = naive::tr_of(Bytes(w.pk(k).begin(), w.pk(k).end()));
        wrong += !std::equal(want.begin(), want.end(),
                             tr.begin() + static_cast<std::ptrdiff_t>(k * ppq::TR_BYTES));
    }

    const auto mu = work.mu(count);
    for (std::size_t i = 0; i < count; ++i) {
        const Bytes want = expected_mu(w, first + i);
        wrong += !std::equal(want.begin(), want.end(),
                             mu.begin() + static_cast<std::ptrdiff_t>(i * ppq::MU_BYTES));
    }
    return wrong;
}

ppq::Workload ragged(const ppq::ParamSet &p, std::uint64_t seed) {
    ppq::GenSpec s;
    s.params = p;
    s.signatures = 24;
    s.keys = 3;
    s.msg_min = 0;
    s.msg_max = 300;
    s.ctx_max = 255;
    s.seed = seed;
    s.threads = 2;
    return ppq::generate(s);
}

TEST_CASE(generated_workloads) {
    for (const ppq::ParamSet &p : ppq::PARAM_SETS) {
        const ppq::Workload w = ragged(p, 500 + static_cast<std::uint64_t>(p.id));
        for (const unsigned threads : {1u, 32u, 256u}) {
            CHECK(check_mu(w, 0, w.sig_count(), threads) == 0);
        }
    }
}

TEST_CASE(acvp_every_interface) {
    for (const ppq::AcvpWorkload &g :
         ppq::load_acvp_sigver(TEST_VECTORS_DIR "/acvp/mldsa_sigver.txt")) {
        CHECK(check_mu(g.workload, 0, g.workload.sig_count(), 64) == 0);
    }
}

TEST_CASE(chunks_use_rebased_offsets) {
    const ppq::Workload w = ragged(ppq::ML_DSA_87, 600);
    CHECK(check_mu(w, 9, 11, 32) == 0);
    CHECK(check_mu(w, 23, 1, 32) == 0);
}

} // namespace

int main() { return gpu_test::run_all(); }
