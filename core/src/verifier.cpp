#include "ppq/verifier.h"

#include <stdexcept>

namespace ppq {

Agreement compare(const Workload &w, const VerifyResult &r) {
    const std::size_t n = w.sig_count();
    if (r.valid.size() != n || (!r.verdicts.empty() && r.verdicts.size() != n)) {
        throw std::invalid_argument("verification result does not cover the workload");
    }

    Agreement a;
    a.checked = n;
    for (std::size_t i = 0; i < n; ++i) {
        const bool valid = r.valid[i] != 0;
        a.valid += valid;

        bool wrong = false;
        if (valid != (w.expected[i] == Verdict::Pass)) {
            ++a.wrong_validity;
            wrong = true;
        }
        if (!r.verdicts.empty() && r.verdicts[i] != w.expected[i]) {
            ++a.wrong_verdict;
            wrong = true;
        }
        if (wrong) {
            ++a.disagreeing;
            if (a.examples.size() < MAX_AGREEMENT_EXAMPLES) {
                a.examples.push_back(i);
            }
        }
    }
    return a;
}

} // namespace ppq
