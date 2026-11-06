#ifndef PPQ_GENERATE_H_
#define PPQ_GENERATE_H_

/// Reproducible verification workloads built with the CPU reference, and the multithreaded
/// CPU verification baseline.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ppq/cpuref.h"
#include "ppq/workload.h"

namespace ppq {

struct GenSpec {
    ParamSet params = ML_DSA_44;
    std::size_t signatures = 1024;
    /// Distinct public keys; signature i uses key i % keys. 0 means one key per signature.
    std::size_t keys = 0;
    /// Fraction of signatures that verify; the rest are tampered
    double pass_rate = 1.0;
    /// Tampering applied to failing signatures, cycled in order (failure k uses
    /// fail_modes[k % size]). Must not contain Tamper::None.
    std::vector<Tamper> fail_modes{Tamper::Message};
    /// Message lengths are uniform in [msg_min, msg_max]
    std::size_t msg_min = 32;
    std::size_t msg_max = 32;
    /// Context lengths are uniform in [0, ctx_max]; at most 255
    std::size_t ctx_max = 0;
    /// Sign with all-zero randomness (the deterministic variant) instead of derived rnd
    bool deterministic = false;
    std::uint64_t seed = 1;
    /// 0 = all hardware threads. The output does not depend on this.
    unsigned threads = 0;
    /// Implementation used for key generation and signing
    cpuref::Impl impl = cpuref::Impl::Ref;
};

/// Throws std::invalid_argument for an inconsistent spec
void validate_spec(const GenSpec &spec);

/// The same spec (apart from threads and impl) always produces the same workload
Workload generate(const GenSpec &spec);

/// Verifies every signature with the CPU reference; 1 = valid. 0 threads = all.
std::vector<std::uint8_t> cpu_verify_batch(const Workload &w, cpuref::Impl impl,
                                           unsigned threads = 0);

} // namespace ppq

#endif
