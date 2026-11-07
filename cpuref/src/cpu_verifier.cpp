#include "ppq/cpu_verifier.h"

#include <chrono>
#include <stdexcept>

#include "ppq/cpuref_stages.h"
#include "ppq/generate.h"
#include "ppq/parallel.h"

namespace ppq {
namespace {

const char *impl_description(cpuref::Impl impl) {
    switch (impl) {
    case cpuref::Impl::Ref:
        return "pq-crystals ref";
    case cpuref::Impl::Avx2:
        return "pq-crystals AVX2";
    case cpuref::Impl::OpenSsl:
        return "OpenSSL";
    }
    return "unknown";
}

/// The verdict for one signature, from the pq-crystals ref stage internals
Verdict stage_verdict(const Workload &w, std::size_t i) {
    cpuref::Bytes mu;
    switch (w.iface) {
    case Interface::External:
        mu = cpuref::mu_external(w.params, w.pk_of(i), w.msg(i), w.ctx(i));
        break;
    case Interface::Internal:
        mu = cpuref::mu_internal(w.params, w.pk_of(i), w.msg(i));
        break;
    case Interface::Mu:
        mu.assign(w.msg(i).begin(), w.msg(i).end());
        break;
    }
    return cpuref::sig_stages(w.params, w.pk_of(i), mu, w.sig(i)).verdict;
}

} // namespace

CpuVerifier::CpuVerifier(const CpuVerifierOptions &options)
    : _options(options), _threads(resolve_threads(options.threads)) {
    if (!cpuref::available(options.impl)) {
        throw std::invalid_argument(std::string("CPU implementation '") +
                                    cpuref::impl_name(options.impl) + "' is not available");
    }
    if (options.verdicts && options.impl != cpuref::Impl::Ref) {
        throw std::invalid_argument("per-signature verdicts come from the pq-crystals ref "
                                    "stages; use the ref implementation");
    }
}

std::string CpuVerifier::name() const {
    return std::string("cpu-") + cpuref::impl_name(_options.impl) +
           (_options.verdicts ? "-verdicts" : "");
}

std::string CpuVerifier::describe() const {
    return std::string("CPU: ") + impl_description(_options.impl) +
           (_options.verdicts ? " stages (with verdicts)" : "") + ", " + std::to_string(_threads) +
           (_threads == 1 ? " thread" : " threads");
}

bool CpuVerifier::supports(Interface iface) const {
    return _options.verdicts || cpuref::supports(_options.impl, iface);
}

VerifyResult CpuVerifier::verify(const Workload &w) {
    if (!supports(w.iface)) {
        throw std::invalid_argument(name() + " cannot verify " + interface_name(w.iface) +
                                    " workloads");
    }

    VerifyResult r;
    const auto start = std::chrono::steady_clock::now();

    if (_options.verdicts) {
        r.verdicts.resize(w.sig_count());
        r.valid.resize(w.sig_count());
        parallel_for(w.sig_count(), _threads, [&](std::size_t i) {
            r.verdicts[i] = stage_verdict(w, i);
            r.valid[i] = r.verdicts[i] == Verdict::Pass ? 1 : 0;
        });
    } else {
        r.valid = cpu_verify_batch(w, _options.impl, _threads);
    }

    r.elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return r;
}

} // namespace ppq
