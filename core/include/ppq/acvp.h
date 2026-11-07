#ifndef PPQ_ACVP_H_
#define PPQ_ACVP_H_

/// NIST ACVP sigVer vectors (tests/vectors/acvp/mldsa_sigver.txt, from tools/acvp.py) as
/// workloads, so any verifier backend can be run against them.

#include <cstdint>
#include <string>
#include <vector>

#include "ppq/workload.h"

namespace ppq {

struct AcvpWorkload {
    Workload workload;
    /// ACVP test case id of each signature
    std::vector<std::int64_t> tc_ids;
};

/// One workload per (parameter set, interface), in order of first appearance. Public keys are
/// deduplicated into each workload's key table; expected verdicts come from the file.
/// Throws std::runtime_error on a missing or malformed file.
std::vector<AcvpWorkload> load_acvp_sigver(const std::string &path);

} // namespace ppq

#endif
