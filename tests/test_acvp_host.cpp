/// NIST ACVP ML-DSA vectors (tests/vectors/acvp): every available CPU implementation must
/// pass them, and the recorded verdicts must agree with pq-crystals and the naive verifier.

#include <iostream>
#include <map>
#include <string>

#include "check.h"
#include "naive_mldsa.h"
#include "ppq/acvp.h"
#include "ppq/cpuref_stages.h"
#include "ppq/generate.h"
#include "vector_util.h"

namespace {

using ppq::cpuref::Impl;
using ppq::cpuref::Interface;
using vector_util::params_of;

const std::vector<ppq::KvRecord> &sigver() {
    static const std::vector<ppq::KvRecord> r =
        ppq::load_kv_records(TEST_VECTORS_DIR "/acvp/mldsa_sigver.txt");
    return r;
}

const std::vector<ppq::KvRecord> &keygen() {
    static const std::vector<ppq::KvRecord> r =
        ppq::load_kv_records(TEST_VECTORS_DIR "/acvp/mldsa_keygen.txt");
    return r;
}

Interface interface_of(const ppq::KvRecord &r) {
    const std::string &s = r.str("interface");
    if (s == "external") {
        return Interface::External;
    }
    if (s == "internal") {
        return Interface::Internal;
    }
    if (s == "mu") {
        return Interface::Mu;
    }
    throw std::runtime_error(r.where() + ": unknown interface '" + s + "'");
}

bool cpu_verify(const ppq::KvRecord &r, Impl impl) {
    const ppq::ParamSet &p = params_of(r);
    const auto pk = r.bytes("pk");
    const auto sig = r.bytes("sig");
    switch (interface_of(r)) {
    case Interface::External:
        return ppq::cpuref::verify(p, pk, r.bytes("msg"), r.bytes("ctx"), sig, impl);
    case Interface::Internal:
        return ppq::cpuref::verify_internal(p, pk, r.bytes("msg"), sig, impl);
    case Interface::Mu:
        return ppq::cpuref::verify_mu(p, pk, r.bytes("mu"), sig, impl);
    }
    return false;
}

/// mu two ways (pq-crystals and naive), which must agree
naive::Bytes mu_of(const ppq::KvRecord &r) {
    const ppq::ParamSet &p = params_of(r);
    const auto pk = r.bytes("pk");
    switch (interface_of(r)) {
    case Interface::External: {
        const auto mu = ppq::cpuref::mu_external(p, pk, r.bytes("msg"), r.bytes("ctx"));
        CHECK(mu == naive::mu_external(pk, r.bytes("msg"), r.bytes("ctx")));
        return mu;
    }
    case Interface::Internal: {
        const auto mu = ppq::cpuref::mu_internal(p, pk, r.bytes("msg"));
        CHECK(mu == naive::shake256(naive::concat({naive::tr_of(pk), r.bytes("msg")}), 64));
        return mu;
    }
    case Interface::Mu:
        return r.bytes("mu");
    }
    return {};
}

TEST_CASE(sigver_corpus_covers_each_interface_and_set) {
    std::map<std::pair<std::string, int>, int> counts;
    for (const ppq::KvRecord &r : sigver()) {
        ++counts[{r.str("interface"), params_of(r).id}];
    }
    CHECK(sigver().size() == 135);
    for (const char *iface : {"external", "internal", "mu"}) {
        for (const ppq::ParamSet &p : ppq::PARAM_SETS) {
            CHECK((counts[{iface, p.id}] == 15));
        }
    }
}

TEST_CASE(every_cpu_implementation_passes_sigver) {
    for (const Impl impl : ppq::cpuref::available_impls()) {
        std::size_t checked = 0;
        std::size_t wrong = 0;
        for (const ppq::KvRecord &r : sigver()) {
            if (!ppq::cpuref::supports(impl, interface_of(r))) {
                continue;
            }
            ++checked;
            if (cpu_verify(r, impl) != (r.integer("passed") == 1)) {
                std::cerr << "    " << ppq::cpuref::impl_name(impl) << " disagrees on " << r.where()
                          << " (tcId " << r.str("tcId") << ")\n";
                ++wrong;
            }
        }
        std::cout << "    " << ppq::cpuref::impl_name(impl) << ": " << checked << " vectors\n";
        CHECK(checked >= 90);
        CHECK(wrong == 0);
    }
}

TEST_CASE(verdicts_agree_with_pq_crystals_and_naive) {
    for (const ppq::KvRecord &r : sigver()) {
        const ppq::ParamSet &p = params_of(r);
        const auto pk = r.bytes("pk");
        const auto sig = r.bytes("sig");
        const auto mu = mu_of(r);
        const ppq::Verdict recorded = vector_util::parse_verdict(r);

        CHECK(ppq::cpuref::sig_stages(p, pk, mu, sig).verdict == recorded);
        CHECK(naive::verify_stages(p, pk, mu, sig).verdict == recorded);
        CHECK((recorded == ppq::Verdict::Pass) == (r.integer("passed") == 1));
    }
}

/// The same vectors as workloads: what `ppq kat` feeds a verifier backend
TEST_CASE(sigver_loads_as_workloads_and_verifies) {
    const auto groups = ppq::load_acvp_sigver(TEST_VECTORS_DIR "/acvp/mldsa_sigver.txt");
    CHECK(groups.size() == 9);

    std::size_t total = 0;
    for (const ppq::AcvpWorkload &g : groups) {
        const ppq::Workload &w = g.workload;
        total += w.sig_count();
        CHECK(w.sig_count() == 15);
        CHECK(g.tc_ids.size() == 15);
        CHECK(w.key_count() >= 1 && w.key_count() <= 15);
        CHECK(w.pass_count() == 3);

        for (const Impl impl : ppq::cpuref::available_impls()) {
            if (!ppq::cpuref::supports(impl, w.iface)) {
                continue;
            }
            const auto ok = ppq::cpu_verify_batch(w, impl, 2);
            for (std::size_t i = 0; i < w.sig_count(); ++i) {
                CHECK((ok[i] == 1) == (w.expected[i] == ppq::Verdict::Pass));
            }
        }
    }
    CHECK(total == sigver().size());
}

TEST_CASE(keygen_matches_acvp) {
    CHECK(keygen().size() == 75);
    for (const Impl impl : ppq::cpuref::available_impls()) {
        std::size_t wrong = 0;
        for (const ppq::KvRecord &r : keygen()) {
            const auto kp = ppq::cpuref::keygen(params_of(r), r.bytes("seed"), impl);
            wrong += kp.pk != r.bytes("pk") || kp.sk != r.bytes("sk");
        }
        CHECK(wrong == 0);
    }
}

} // namespace

int main() { return check::run_all(); }
