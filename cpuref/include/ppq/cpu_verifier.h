#ifndef PPQ_CPU_VERIFIER_H_
#define PPQ_CPU_VERIFIER_H_

/// The CPU verification backend: the third-party CPU reference behind the Verifier interface.

#include <string>

#include "ppq/cpuref.h"
#include "ppq/verifier.h"

namespace ppq {

struct CpuVerifierOptions {
    cpuref::Impl impl = cpuref::Impl::Ref;
    /// 0 = all hardware threads
    unsigned threads = 0;
    /// Report a verdict per signature by running the pq-crystals ref stages (slower; Ref only)
    bool verdicts = false;
};

class CpuVerifier final : public Verifier {
  public:
    /// Throws std::invalid_argument if the impl is unavailable, or verdicts are asked of a
    /// non-Ref impl
    explicit CpuVerifier(const CpuVerifierOptions &options);

    std::string name() const override;
    std::string describe() const override;
    bool supports(Interface iface) const override;
    bool reports_verdicts() const override { return _options.verdicts; }
    VerifyResult verify(const Workload &w) override;

    unsigned threads() const { return _threads; }

  private:
    CpuVerifierOptions _options;
    unsigned _threads;
};

} // namespace ppq

#endif
