#ifndef PPQ_GPU_VERIFIER_H_
#define PPQ_GPU_VERIFIER_H_

/// The staged GPU verifier: every stage of ML-DSA.Verify as its own kernel, run chunk by
/// chunk on the current device behind the Verifier interface.

#include <cstddef>
#include <memory>
#include <string>

#include "ppq/verifier.h"

namespace ppq {

struct GpuVerifierOptions {
    /// Threads per block for every stage kernel
    unsigned threads = 128;
    /// Threads per polynomial in the cooperative NTTs; `threads` must be a multiple of it
    unsigned ntt_group = 32;
    /// Signatures per chunk: bounds device memory (about 20 KiB per ML-DSA-87 signature, plus
    /// up to 64 KiB per cached key)
    std::size_t chunk = 8192;
};

class GpuVerifier final : public Verifier {
  public:
    /// Throws std::invalid_argument for inconsistent options
    explicit GpuVerifier(const GpuVerifierOptions &options = {});
    ~GpuVerifier() override;

    GpuVerifier(const GpuVerifier &) = delete;
    GpuVerifier &operator=(const GpuVerifier &) = delete;

    std::string name() const override { return "gpu-staged"; }
    std::string describe() const override;
    bool supports(Interface) const override { return true; }
    bool reports_verdicts() const override { return true; }

    /// Per-stage times are reported in VerifyResult::stages, summed over chunks
    VerifyResult verify(const Workload &w) override;

    const GpuVerifierOptions &options() const { return _options; }

  private:
    struct State;
    GpuVerifierOptions _options;
    std::unique_ptr<State> _state;
};

} // namespace ppq

#endif
