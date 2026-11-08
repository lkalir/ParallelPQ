/// ppq.exe's option parsing and commands, driven in-process

#include <initializer_list>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "check.h"
#include "commands.h"
#include "options.h"
#include "temp_dir.h"

namespace {

using ppq::app::Command;
using ppq::app::Options;
using ppq::app::UsageError;

using Json = nlohmann::json;

Options parse(std::vector<std::string> args) {
    args.insert(args.begin(), "ppq.exe");
    std::vector<const char *> argv;
    for (const std::string &a : args) {
        argv.push_back(a.c_str());
    }
    argv.push_back(nullptr);
    return ppq::app::parse_options(static_cast<int>(args.size()), argv.data());
}

/// True when parsing throws UsageError and the message contains every needle
bool usage_error(std::vector<std::string> args, std::initializer_list<std::string> needles) {
    try {
        parse(std::move(args));
    } catch (const UsageError &e) {
        const std::string what = e.what();
        for (const std::string &needle : needles) {
            if (what.find(needle) == std::string::npos) {
                std::cerr << "    message was: " << what << "\n";
                return false;
            }
        }
        return true;
    }
    return false;
}

bool usage_error(std::vector<std::string> args, const std::string &needle) {
    return usage_error(std::move(args), {needle});
}

struct Run {
    int status;
    std::string out;
};

Run run(std::vector<std::string> args) {
    const Options o = parse(std::move(args));
    std::ostringstream out;
    std::ostringstream err;
    ppq::app::Context ctx{out, err, nullptr};
    const int status = ppq::app::run_command(o, ctx);
    return {status, out.str()};
}

bool contains(const std::string &s, const std::string &needle) {
    if (s.find(needle) == std::string::npos) {
        std::cerr << "    missing '" << needle << "' in:\n" << s;
        return false;
    }
    return true;
}

TEST_CASE(defaults_and_values_parse) {
    const Options d = parse({"verify"});
    CHECK(d.command == Command::Verify);
    CHECK(d.spec.params.id == 44);
    CHECK(d.spec.signatures == 1024);
    CHECK(d.spec.pass_rate == 1.0);
    CHECK(d.repeat == 1 && d.warmup == 0 && d.threads == 0);
    CHECK(!d.cpu_impl.has_value());
    CHECK(d.format == ppq::app::Format::Text);

    const Options o = parse(
        {"verify",     "-p",          "ML-DSA-87", "-n",       "64",          "-k",
         "4",          "--pass-rate", "0.25",      "--tamper", "hint,z-norm", "--msg-len",
         "5:9",        "--ctx-len",   "12",        "--seed",   "77",          "--deterministic",
         "-t",         "3",           "-r",        "4",        "--warmup",    "2",
         "--cpu-impl", "ref",         "--format",  "json"});
    CHECK(o.spec.params.id == 87);
    CHECK(o.spec.signatures == 64 && o.spec.keys == 4);
    CHECK(o.spec.pass_rate == 0.25);
    CHECK((o.spec.fail_modes == std::vector<ppq::Tamper>{ppq::Tamper::Hint, ppq::Tamper::ZNorm}));
    CHECK(o.spec.msg_min == 5 && o.spec.msg_max == 9 && o.spec.ctx_max == 12);
    CHECK(o.spec.seed == 77 && o.spec.deterministic);
    CHECK(o.threads == 3 && o.spec.threads == 3);
    CHECK(o.repeat == 4 && o.warmup == 2);
    CHECK(o.cpu_impl == ppq::cpuref::Impl::Ref);
    CHECK(o.format == ppq::app::Format::Json);
    CHECK(o.workload_options);

    CHECK(parse({"verify", "--tamper", "mixed"}).spec.fail_modes.size() == 4);
    CHECK(parse({"--help"}).command == Command::Help);
    CHECK(parse({"kat", "-V"}).command == Command::Version);
    CHECK(parse({"generate", "-o", "x.ppq"}).output == "x.ppq");
    CHECK(parse({"verify", "-i", "x.ppq"}).input == "x.ppq");
}

TEST_CASE(bad_command_lines_are_usage_errors) {
    CHECK(usage_error({}, "no command given"));
    CHECK(usage_error({"frobnicate"}, "not expected: frobnicate"));
    CHECK(usage_error({"verify", "extra"}, "not expected: extra"));
    CHECK(usage_error({"verify", "--bogus"}, "not expected: --bogus"));
    CHECK(usage_error({"verify", "-n"}, "--sigs: 1 required"));
    CHECK(usage_error({"verify", "-n", "12x"}, "--sigs"));
    CHECK(usage_error({"verify", "-n", "-3"}, "--sigs"));
    CHECK(usage_error({"verify", "-n", "268435457"}, "not in range"));
    CHECK(usage_error({"verify", "-p", "50"}, "44, 65 or 87"));
    CHECK(usage_error({"verify", "--pass-rate", "1.5"}, "not in range"));
    CHECK(usage_error({"verify", "--pass-rate", "nan"}, "[0, 1]"));
    CHECK(usage_error({"verify", "--tamper", "none"}, "--tamper"));
    CHECK(usage_error({"verify", "--msg-len", "9:5"}, "MIN <= MAX"));
    CHECK(usage_error({"verify", "--msg-len", "x"}, "--msg-len"));
    CHECK(usage_error({"verify", "--ctx-len", "256"}, "not in range"));
    CHECK(usage_error({"verify", "-r", "0"}, "not in range"));
    CHECK(usage_error({"verify", "--format", "xml"}, "xml not in"));
    CHECK(usage_error({"verify", "--cpu-impl", "fast"}, "fast not in"));
    CHECK(usage_error({"verify", "--backend", "gpu"}, "not implemented yet"));
    for (const ppq::cpuref::Impl impl : ppq::cpuref::available_impls()) {
        if (impl != ppq::cpuref::Impl::Ref) {
            CHECK(usage_error({"verify", "--verdicts", "--cpu-impl", ppq::cpuref::impl_name(impl)},
                              "--verdicts"));
        }
    }
    for (const ppq::cpuref::Impl impl : ppq::cpuref::ALL_IMPLS) {
        if (!ppq::cpuref::available(impl)) {
            CHECK(usage_error({"verify", "--cpu-impl", ppq::cpuref::impl_name(impl)},
                              "not available"));
        }
    }
    CHECK(usage_error({"generate"}, "--output is required"));
    // options of another command
    // (CLI11 releases list the leftover arguments in different orders)
    CHECK(usage_error({"generate", "-o", "x", "-r", "2"}, {"not expected", "-r"}));
    CHECK(usage_error({"kat", "-n", "5"}, {"not expected", "-n"}));
    CHECK(usage_error({"info", "--verdicts"}, "not expected: --verdicts"));
    CHECK(usage_error({"verify", "-i", "x.ppq", "--seed", "3"}, "excludes --seed"));
    CHECK(usage_error({"verify", "-n", "4", "-k", "5"}, "more keys than signatures"));
}

TEST_CASE(verify_generated_workloads) {
    const Run r =
        run({"verify", "-p", "65", "-n", "40", "-k", "3", "--pass-rate", "0.5", "--tamper", "mixed",
             "--msg-len", "0:50", "--ctx-len", "9", "-t", "2", "-r", "2"});
    CHECK(r.status == 0);
    CHECK(contains(r.out, "ML-DSA-65, external, 40 signatures, 3 keys, 20 expected to pass"));
    CHECK(contains(r.out, "20 valid, 20 invalid"));
    CHECK(contains(r.out, "all 40 agree"));

    const Run v = run(
        {"verify", "-n", "20", "--pass-rate", "0.5", "--tamper", "mixed", "--verdicts", "-t", "2"});
    CHECK(v.status == 0);
    CHECK(contains(v.out, "verdicts  pass 10, bad-commitment 6, z-norm 2, bad-hint 2"));
}

TEST_CASE(generate_then_verify_through_a_file) {
    const test_util::TempDir dir("app");
    const std::string path = dir.file("w.ppq");
    const Run g = run({"generate", "-o", path, "-p", "87", "-n", "12", "-k", "1", "--pass-rate",
                       "0.5", "--tamper", "hint"});
    CHECK(g.status == 0);
    CHECK(contains(g.out, "12 signatures, 1 key,"));

    const Run v = run({"verify", "-i", path, "--format", "json"});
    CHECK(v.status == 0);
    const Json j = Json::parse(v.out);
    CHECK(j["command"] == "verify");
    CHECK(j["workload"]["source"] == "file");
    CHECK(j["workload"]["path"] == path);
    CHECK(j["workload"]["params"] == "ML-DSA-87");
    CHECK(j["workload"]["signatures"] == 12 && j["workload"]["expected_pass"] == 6);
    CHECK(j["result"]["valid"] == 6 && j["result"]["invalid"] == 6);
    CHECK(j["runs"]["ms"].size() == 1);
    CHECK(j["check"]["ok"] == true);
    CHECK(j["check"]["examples"].empty());

    // --format may also come before the command
    CHECK(Json::parse(run({"--format", "json", "verify", "-i", path}).out)["check"]["ok"] == true);
}

TEST_CASE(disagreement_with_expected_verdicts_fails) {
    const test_util::TempDir dir("app");
    const std::string path = dir.file("wrong.ppq");
    run({"generate", "-o", path, "-n", "6", "--pass-rate", "1"});

    // Claim a valid signature should fail: the verifier must report the disagreement
    ppq::Workload w = ppq::load_workload(path);
    w.expected[3] = ppq::Verdict::BadCommitment;
    ppq::save_workload(w, path);

    const Run r = run({"verify", "-i", path});
    CHECK(r.status == 1);
    CHECK(contains(r.out, "FAILED: 1 wrong validity"));
    CHECK(contains(r.out, "signature 3: expected bad-commitment, got valid"));

    const Run j = run({"verify", "-i", path, "--format", "json"});
    CHECK(j.status == 1);
    const Json report = Json::parse(j.out);
    CHECK(report["check"]["ok"] == false);
    CHECK(report["check"]["wrong_validity"] == 1);
    CHECK((report["check"]["examples"] == Json::array({3})));
}

TEST_CASE(kat_runs_every_acvp_group) {
    const Run r = run({"kat", "-t", "2"});
    CHECK(r.status == 0);
    CHECK(contains(r.out, "135 agree, 0 disagree, 0 skipped"));

    if (ppq::cpuref::available(ppq::cpuref::Impl::Avx2)) {
        const Run a = run({"kat", "--cpu-impl", "avx2"});
        CHECK(a.status == 0);
        CHECK(contains(a.out, "90 agree, 0 disagree, 45 skipped"));
    }

    const Json j = Json::parse(run({"kat", "--format", "json"}).out);
    CHECK(j["agree"] == 135 && j["disagree"] == 0 && j["ok"] == true);
    CHECK(j["groups"].size() == 9);
    for (const Json &g : j["groups"]) {
        CHECK(g["ok"] == true && g["failed_tc_ids"].empty());
    }

    bool threw = false;
    try {
        run({"kat", "--vectors", "/nonexistent"});
    } catch (const std::runtime_error &) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE(info_and_help) {
    const Run i = run({"info"});
    CHECK(i.status == 0);
    CHECK(contains(i.out, "CPU only"));
    CHECK(contains(i.out, "ref (pq-crystals ref)"));

    const Json j = Json::parse(run({"info", "--format", "json"}).out);
    CHECK(j["build"] == "cpu-only");
    CHECK(j["cpu_impls"][0]["name"] == "ref");
    CHECK(j["gpus"].empty());

    const Run h = run({"--help"});
    CHECK(h.status == 0);
    CHECK(contains(h.out, "Usage: ppq.exe"));
    CHECK(contains(h.out, "generate"));
    CHECK(contains(h.out, "Exit status"));
    // a command's help lists its own options
    const Run vh = run({"verify", "--help"});
    CHECK(contains(vh.out, "--pass-rate"));
    CHECK(contains(vh.out, "--repeat"));
    CHECK(run({"generate", "-h"}).out.find("--repeat") == std::string::npos);
    CHECK(contains(run({"--help-all"}).out, "--vectors"));
    CHECK(contains(run({"-V"}).out, "ppq "));
}

} // namespace

int main() { return check::run_all(); }
