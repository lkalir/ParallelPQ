/// The CPU backend behind the Verifier interface, and compare() against ground truth

#include <algorithm>
#include <stdexcept>
#include <string>

#include "check.h"
#include "ppq/acvp.h"
#include "ppq/cpu_verifier.h"
#include "ppq/generate.h"

namespace {

using ppq::cpuref::Impl;

ppq::Workload mixed(const ppq::ParamSet &p, std::size_t n) {
    ppq::GenSpec s;
    s.params = p;
    s.signatures = n;
    s.keys = 3;
    s.pass_rate = 0.5;
    s.fail_modes = {ppq::Tamper::Message, ppq::Tamper::Commitment, ppq::Tamper::ZNorm,
                    ppq::Tamper::Hint};
    s.msg_min = 0;
    s.msg_max = 100;
    s.ctx_max = 20;
    s.seed = 99;
    s.threads = 2;
    return ppq::generate(s);
}

template <class Fn> bool throws_invalid(Fn &&fn) {
    try {
        fn();
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

TEST_CASE(every_impl_agrees_with_ground_truth) {
    for (const ppq::ParamSet &p : ppq::PARAM_SETS) {
        const ppq::Workload w = mixed(p, 40);
        for (const Impl impl : ppq::cpuref::available_impls()) {
            ppq::CpuVerifier v({impl, 2, false});
            const ppq::VerifyResult r = v.verify(w);
            const ppq::Agreement a = ppq::compare(w, r);
            CHECK(a.ok());
            CHECK(a.checked == 40);
            CHECK(a.valid == 20);
            CHECK(r.verdicts.empty());
            CHECK(r.elapsed_ms > 0.0);
        }
    }
}

TEST_CASE(verdict_mode_reports_the_reason) {
    const ppq::Workload w = mixed(ppq::ML_DSA_65, 40);
    ppq::CpuVerifier v({Impl::Ref, 2, true});
    CHECK(v.reports_verdicts());
    const ppq::VerifyResult r = v.verify(w);
    CHECK(r.verdicts == w.expected);
    CHECK(ppq::compare(w, r).ok());
}

TEST_CASE(acvp_workloads_through_the_backend) {
    const auto groups = ppq::load_acvp_sigver(TEST_VECTORS_DIR "/acvp/mldsa_sigver.txt");
    for (const Impl impl : ppq::cpuref::available_impls()) {
        ppq::CpuVerifier v({impl, 0, false});
        for (const ppq::AcvpWorkload &g : groups) {
            if (!v.supports(g.workload.iface)) {
                CHECK(throws_invalid([&] { v.verify(g.workload); }));
                continue;
            }
            CHECK(ppq::compare(g.workload, v.verify(g.workload)).ok());
        }
    }

    ppq::CpuVerifier verdicts({Impl::Ref, 0, true});
    for (const ppq::AcvpWorkload &g : groups) {
        CHECK(ppq::compare(g.workload, verdicts.verify(g.workload)).ok());
    }
}

TEST_CASE(compare_reports_disagreements) {
    ppq::Workload w = mixed(ppq::ML_DSA_44, 20);
    ppq::CpuVerifier v({Impl::Ref, 1, true});
    const ppq::VerifyResult r = v.verify(w);

    // A wrong expectation in either direction is caught, and so is a wrong reason
    const std::size_t pass_at = static_cast<std::size_t>(
        std::find(w.expected.begin(), w.expected.end(), ppq::Verdict::Pass) - w.expected.begin());
    const std::size_t fail_at = static_cast<std::size_t>(
        std::find(w.expected.begin(), w.expected.end(), ppq::Verdict::ZNorm) - w.expected.begin());
    w.expected[pass_at] = ppq::Verdict::BadCommitment;
    w.expected[fail_at] = ppq::Verdict::BadHint;

    const ppq::Agreement a = ppq::compare(w, r);
    CHECK(!a.ok());
    CHECK(a.wrong_validity == 1);
    CHECK(a.wrong_verdict == 2);
    CHECK(a.examples.size() == 2);

    ppq::VerifyResult short_result = r;
    short_result.valid.pop_back();
    CHECK(throws_invalid([&] { ppq::compare(w, short_result); }));
}

TEST_CASE(results_do_not_depend_on_thread_count) {
    const ppq::Workload w = mixed(ppq::ML_DSA_87, 30);
    ppq::CpuVerifier one({Impl::Ref, 1, false});
    ppq::CpuVerifier many({Impl::Ref, 8, false});
    CHECK(one.threads() == 1);
    CHECK(many.threads() == 8);
    CHECK(one.verify(w).valid == many.verify(w).valid);
}

TEST_CASE(construction_and_naming) {
    CHECK(ppq::CpuVerifier({Impl::Ref, 4, false}).name() == "cpu-ref");
    CHECK(ppq::CpuVerifier({Impl::Ref, 1, true}).name() == "cpu-ref-verdicts");
    CHECK(ppq::CpuVerifier({Impl::Ref, 1, false}).describe().find("1 thread") != std::string::npos);
    CHECK(throws_invalid([] { ppq::CpuVerifier({Impl::OpenSsl, 1, true}); }));
    for (const Impl impl : ppq::cpuref::ALL_IMPLS) {
        if (!ppq::cpuref::available(impl)) {
            CHECK(throws_invalid([&] { ppq::CpuVerifier({impl, 1, false}); }));
        }
    }

    ppq::Workload empty;
    const ppq::VerifyResult r = ppq::CpuVerifier({Impl::Ref, 0, false}).verify(empty);
    CHECK(r.valid.empty());
    CHECK(ppq::compare(empty, r).ok());
}

} // namespace

int main() { return check::run_all(); }
