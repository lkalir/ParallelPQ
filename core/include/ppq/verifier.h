#ifndef PPQ_VERIFIER_H_
#define PPQ_VERIFIER_H_

/// The interface every verification backend implements (the CPU reference now, the GPU
/// verifier later), and the check of a backend's results against a workload's ground truth.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ppq/workload.h"

namespace ppq {

struct StageTime {
    std::string name;
    double ms = 0.0;
};

struct VerifyResult {
    /// 1 when signature i verified
    std::vector<std::uint8_t> valid;
    /// Why each signature passed or failed; empty unless the backend reports verdicts
    std::vector<Verdict> verdicts;
    /// Wall time of the whole verify() call
    double elapsed_ms = 0.0;
    /// Optional breakdown of elapsed_ms (e.g. copies and kernels)
    std::vector<StageTime> stages;
};

class Verifier {
  public:
    virtual ~Verifier() = default;

    /// Short identifier for reports and file names, e.g. "cpu-avx2"
    virtual std::string name() const = 0;
    /// One line for humans, e.g. "CPU: pq-crystals AVX2, 16 threads"
    virtual std::string describe() const = 0;
    virtual bool supports(Interface iface) const = 0;
    virtual bool reports_verdicts() const = 0;

    /// Throws std::invalid_argument for a workload whose interface is unsupported
    virtual VerifyResult verify(const Workload &w) = 0;
};

/// How a backend's results line up with a workload's expected verdicts
struct Agreement {
    std::size_t checked = 0;
    std::size_t valid = 0;
    /// Signatures whose valid flag disagrees with expected == Pass
    std::size_t wrong_validity = 0;
    /// Signatures whose verdict disagrees (only when the backend reports verdicts)
    std::size_t wrong_verdict = 0;
    /// Signatures with a disagreement of either kind
    std::size_t disagreeing = 0;
    /// Indices of the first few of them
    std::vector<std::size_t> examples;

    bool ok() const { return wrong_validity == 0 && wrong_verdict == 0; }
};

constexpr std::size_t MAX_AGREEMENT_EXAMPLES = 10;

/// Throws std::invalid_argument if the result does not have one entry per signature
Agreement compare(const Workload &w, const VerifyResult &r);

} // namespace ppq

#endif
