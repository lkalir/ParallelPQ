/// Workload generator: ground truth agrees with the CPU reference, output is reproducible,
/// and the spec is honoured.

#include <algorithm>
#include <stdexcept>
#include <string>

#include "check.h"
#include "ppq/generate.h"
#include "temp_dir.h"

namespace {

using ppq::Tamper;
using ppq::cpuref::Impl;

const std::vector<Tamper> ALL_FAIL_MODES = {Tamper::Message, Tamper::Commitment, Tamper::ZNorm,
                                            Tamper::Hint};

ppq::GenSpec mixed_spec(const ppq::ParamSet &p, std::size_t n) {
    ppq::GenSpec s;
    s.params = p;
    s.signatures = n;
    s.keys = 5;
    s.pass_rate = 0.5;
    s.fail_modes = ALL_FAIL_MODES;
    s.msg_min = 0;
    s.msg_max = 200;
    s.ctx_max = 255;
    s.seed = 42;
    s.threads = 2;
    return s;
}

/// Every signature's CPU result must match the generator's expected verdict
std::size_t disagreements(const ppq::Workload &w, Impl impl) {
    const auto ok = ppq::cpu_verify_batch(w, impl, 2);
    std::size_t bad = 0;
    for (std::size_t i = 0; i < w.sig_count(); ++i) {
        const bool want = w.expected[i] == ppq::Verdict::Pass;
        if ((ok[i] != 0) != want) {
            if (bad < 5) {
                std::cerr << "    signature " << i << ": tamper " << ppq::tamper_name(w.tamper[i])
                          << ", CPU says " << (ok[i] ? "valid" : "invalid") << "\n";
            }
            ++bad;
        }
    }
    return bad;
}

bool spec_rejected(const ppq::GenSpec &s) {
    try {
        ppq::validate_spec(s);
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

TEST_CASE(expected_verdicts_match_the_cpu_reference) {
    for (const ppq::ParamSet &p : ppq::PARAM_SETS) {
        const ppq::Workload w = ppq::generate(mixed_spec(p, 80));
        CHECK(w.sig_count() == 80);
        CHECK(w.pass_count() == 40);
        for (const Impl impl : ppq::cpuref::available_impls()) {
            CHECK(disagreements(w, impl) == 0);
        }
    }
}

TEST_CASE(each_tamper_mode_alone_fails_verification) {
    for (const ppq::ParamSet &p : ppq::PARAM_SETS) {
        for (const Tamper t : ALL_FAIL_MODES) {
            ppq::GenSpec s = mixed_spec(p, 24);
            s.pass_rate = 0.0;
            s.fail_modes = {t};
            const ppq::Workload w = ppq::generate(s);
            CHECK(w.pass_count() == 0);
            CHECK(std::all_of(w.tamper.begin(), w.tamper.end(), [&](Tamper x) { return x == t; }));
            CHECK(disagreements(w, Impl::Ref) == 0);
        }
    }
}

TEST_CASE(message_tamper_works_on_empty_messages) {
    ppq::GenSpec s = mixed_spec(ppq::ML_DSA_44, 20);
    s.msg_min = 0;
    s.msg_max = 0;
    s.fail_modes = {Tamper::Message};
    const ppq::Workload w = ppq::generate(s);
    for (std::size_t i = 0; i < w.sig_count(); ++i) {
        CHECK(w.msg(i).size() == (w.tamper[i] == Tamper::Message ? 1u : 0u));
    }
    CHECK(disagreements(w, Impl::Ref) == 0);
}

TEST_CASE(output_is_reproducible_and_thread_independent) {
    ppq::GenSpec s = mixed_spec(ppq::ML_DSA_65, 40);
    s.threads = 1;
    const ppq::Workload a = ppq::generate(s);
    s.threads = 4;
    const ppq::Workload b = ppq::generate(s);
    CHECK(same_workload(a, b));

    if (ppq::cpuref::available(Impl::Avx2)) {
        s.impl = Impl::Avx2;
        CHECK(same_workload(a, ppq::generate(s)));
    }

    s.seed = 43;
    const ppq::Workload c = ppq::generate(s);
    CHECK(a.pks != c.pks);
    CHECK(a.sigs != c.sigs);
}

TEST_CASE(spec_shapes_the_workload) {
    ppq::GenSpec s = mixed_spec(ppq::ML_DSA_87, 30);
    s.keys = 1;
    s.pass_rate = 1.0;
    s.msg_min = 10;
    s.msg_max = 20;
    s.ctx_max = 0;
    const ppq::Workload same_key = ppq::generate(s);
    CHECK(same_key.key_count() == 1);
    CHECK(std::all_of(same_key.key_index.begin(), same_key.key_index.end(),
                      [](std::uint32_t k) { return k == 0; }));
    CHECK(same_key.pass_count() == 30);
    for (std::size_t i = 0; i < same_key.sig_count(); ++i) {
        CHECK(same_key.msg(i).size() >= 10 && same_key.msg(i).size() <= 20);
        CHECK(same_key.ctx(i).empty());
    }

    s.keys = 0;
    const ppq::Workload distinct = ppq::generate(s);
    CHECK(distinct.key_count() == 30);
    CHECK(distinct.key_index[29] == 29);

    s.signatures = 0;
    CHECK(ppq::generate(s).sig_count() == 0);
}

TEST_CASE(deterministic_signing_repeats_signatures) {
    ppq::GenSpec s = mixed_spec(ppq::ML_DSA_44, 10);
    s.pass_rate = 1.0;
    s.deterministic = true;
    const ppq::Workload a = ppq::generate(s);
    s.deterministic = false;
    const ppq::Workload b = ppq::generate(s);
    CHECK(a.pks == b.pks);
    CHECK(a.msg_data == b.msg_data);
    CHECK(a.sigs != b.sigs);
    CHECK(disagreements(a, Impl::Ref) == 0);
}

TEST_CASE(generated_workload_survives_a_file_round_trip) {
    const test_util::TempDir dir("generate");
    const ppq::Workload w = ppq::generate(mixed_spec(ppq::ML_DSA_65, 30));
    const std::string path = dir.file("w.ppq");
    ppq::save_workload(w, path);
    const ppq::Workload back = ppq::load_workload(path);
    CHECK(same_workload(w, back));
    CHECK(back.seed == 42);
    CHECK(disagreements(back, Impl::Ref) == 0);
}

TEST_CASE(bad_specs_are_rejected) {
    const ppq::GenSpec good = mixed_spec(ppq::ML_DSA_44, 10);
    CHECK(!spec_rejected(good));

    ppq::GenSpec s = good;
    s.pass_rate = 1.5;
    CHECK(spec_rejected(s));
    s = good;
    s.keys = 11;
    CHECK(spec_rejected(s));
    s = good;
    s.msg_min = 5;
    s.msg_max = 4;
    CHECK(spec_rejected(s));
    s = good;
    s.ctx_max = 256;
    CHECK(spec_rejected(s));
    s = good;
    s.fail_modes = {};
    CHECK(spec_rejected(s));
    s = good;
    s.fail_modes = {Tamper::None};
    CHECK(spec_rejected(s));
    s = good;
    s.params = ppq::ParamSet{};
    CHECK(spec_rejected(s));
}

} // namespace

int main() { return check::run_all(); }
