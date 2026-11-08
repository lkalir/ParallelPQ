#include "options.h"

#include <CLI/CLI.hpp>

#include <cmath>
#include <map>
#include <sstream>
#include <string_view>
#include <vector>

#ifndef PPQ_VERSION
#define PPQ_VERSION "unknown"
#endif
#ifndef PPQ_VECTORS_DIR
#define PPQ_VECTORS_DIR "tests/vectors"
#endif

namespace ppq::app {
namespace {

constexpr std::uint64_t MAX_SIGS = 1u << 28;
constexpr std::uint64_t MAX_MSG_LEN = 1u << 30;
constexpr const char *WORKLOAD = "Workload";
constexpr const char *VERIFICATION = "Verification";

[[noreturn]] void usage_error(const std::string &what) { throw UsageError(what); }

const ParamSet &parse_params(std::string_view s) {
    if (s.substr(0, 7) == "ML-DSA-" || s.substr(0, 7) == "ml-dsa-") {
        s.remove_prefix(7);
    }
    for (const ParamSet &p : PARAM_SETS) {
        if (s == std::to_string(p.id)) {
            return p;
        }
    }
    usage_error("--params: expected 44, 65 or 87, got '" + std::string(s) + "'");
}

std::vector<Tamper> parse_tampers(const std::string &s) {
    if (s == "mixed") {
        return {Tamper::Message, Tamper::Commitment, Tamper::ZNorm, Tamper::Hint};
    }
    std::vector<Tamper> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        const auto t = parse_tamper(item);
        if (!t || *t == Tamper::None) {
            usage_error("--tamper: expected message, commitment, z-norm, hint (comma separated) "
                        "or mixed, got '" +
                        item + "'");
        }
        out.push_back(*t);
    }
    if (out.empty()) {
        usage_error("--tamper: needs at least one mode");
    }
    return out;
}

std::uint64_t parse_len(const std::string &s) {
    std::uint64_t v = 0;
    if (s.empty() || s.front() == '-' || !CLI::detail::lexical_cast(s, v) || v > MAX_MSG_LEN) {
        usage_error("--msg-len: expected a whole number up to " + std::to_string(MAX_MSG_LEN) +
                    ", got '" + s + "'");
    }
    return v;
}

void parse_msg_len(const std::string &s, GenSpec &spec) {
    const auto colon = s.find(':');
    if (colon == std::string::npos) {
        spec.msg_min = spec.msg_max = parse_len(s);
        return;
    }
    spec.msg_min = parse_len(s.substr(0, colon));
    spec.msg_max = parse_len(s.substr(colon + 1));
    if (spec.msg_min > spec.msg_max) {
        usage_error("--msg-len: MIN:MAX needs MIN <= MAX");
    }
}

/// `s` already passed the IsMember check
std::optional<cpuref::Impl> parse_impl(const char *opt, const std::string &s) {
    if (s == "auto") {
        return std::nullopt;
    }
    for (const cpuref::Impl impl : cpuref::ALL_IMPLS) {
        if (s == cpuref::impl_name(impl)) {
            if (!cpuref::available(impl)) {
                usage_error(std::string(opt) + ": " + s +
                            " is not available in this build or on this CPU");
            }
            return impl;
        }
    }
    usage_error(std::string(opt) + ": unknown implementation '" + s + "'");
}

std::vector<std::string> impl_choices() {
    std::vector<std::string> out{"auto"};
    for (const cpuref::Impl impl : cpuref::ALL_IMPLS) {
        out.emplace_back(cpuref::impl_name(impl));
    }
    return out;
}

/// CLI11's formatter, without the exclusion lists that --input would add to every workload
/// option
class Formatter : public CLI::Formatter {
  public:
    std::string make_option_opts(const CLI::Option *opt) const override {
        std::string out;
        if (opt->get_type_size() != 0) {
            out += " " + opt->get_type_name();
            if (!opt->get_default_str().empty()) {
                out += " [" + opt->get_default_str() + "]";
            }
            if (opt->get_required()) {
                out += " " + get_label("REQUIRED");
            }
        }
        return out;
    }
};

/// A validator that leaves the option's type name alone in the help
CLI::Validator quiet(CLI::Validator v) { return v.description(""); }

std::string impl_type() {
    std::string out;
    for (const std::string &c : impl_choices()) {
        out += (out.empty() ? "" : "|") + c;
    }
    return out;
}

/// The command line: CLI11 binds what it can straight into Options; the values with their
/// own syntax are kept as strings and converted once parsing succeeds.
class Cli {
  public:
    explicit Cli(Options &o)
        : _o(o), _app("ParallelPQ: batched ML-DSA (FIPS 204) signature verification", "ppq.exe") {
        _o.vectors_dir = PPQ_VECTORS_DIR;

        _app.formatter(std::make_shared<Formatter>());
        _app.usage("Usage: ppq.exe [--format text|json] <command> [options]");
        _app.require_subcommand(0, 1);
        _app.set_help_all_flag("--help-all", "Print help for every command");
        _app.set_version_flag("-V,--version", "ppq " PPQ_VERSION);
        _app.add_option("--format", _format, "Report format")
            ->type_name("text|json")
            ->check(quiet(CLI::IsMember({"text", "json"})))
            ->capture_default_str();
        _app.footer("Exit status: 0 success, 1 failure (including results that disagree with "
                    "the expected verdicts), 2 usage error.");

        CLI::App *info = _app.add_subcommand("info", "CPU implementations, threads and GPUs");
        _sub[Command::Info] = info;

        CLI::App *gen =
            _app.add_subcommand("generate", "Write a workload of keys and signatures to a file");
        _sub[Command::Generate] = gen;
        gen->add_option("-o,--output", _o.output, "Where to write the workload (.ppq)")
            ->type_name("FILE")
            ->required();
        workload_options(gen);
        threads_option(gen);

        CLI::App *verify = _app.add_subcommand(
            "verify", "Verify a generated or loaded workload and check the results");
        _sub[Command::Verify] = verify;
        CLI::Option *input =
            verify->add_option("-i,--input", _o.input, "Load this workload instead of generating")
                ->type_name("FILE");
        for (CLI::Option *w : workload_options(verify)) {
            input->excludes(w);
        }
        verifier_options(verify);
        verify->add_option("-r,--repeat", _o.repeat, "Timed runs")
            ->type_name("N")
            ->check(quiet(CLI::Range(1u, 1000000u)))
            ->capture_default_str()
            ->group(VERIFICATION);
        verify->add_option("--warmup", _o.warmup, "Untimed runs first")
            ->type_name("N")
            ->check(quiet(CLI::Range(0u, 1000000u)))
            ->capture_default_str()
            ->group(VERIFICATION);

        CLI::App *kat = _app.add_subcommand("kat", "Verify the NIST ACVP sigVer vectors");
        _sub[Command::Kat] = kat;
        verifier_options(kat);
        kat->add_option("--vectors", _o.vectors_dir, "Vector directory")
            ->type_name("DIR")
            ->capture_default_str();

        for (auto &[command, sub] : _sub) {
            sub->fallthrough(); // --format, --version after the command
            sub->usage("Usage: ppq.exe " + sub->get_name() + " [options]");
        }
    }

    void parse(int argc, const char *const *argv) {
        try {
            _app.parse(argc, argv);
        } catch (const CLI::CallForHelp &e) {
            help(e);
            return;
        } catch (const CLI::CallForAllHelp &e) {
            help(e);
            return;
        } catch (const CLI::CallForVersion &e) {
            _o.command = Command::Version;
            _o.version_text = e.what();
            return;
        } catch (const CLI::ParseError &e) {
            // Errors from the root app carry its name, which main already prints as "ppq: "
            std::string_view what = e.what();
            const std::string_view prefix = "ppq.exe: ";
            if (what.substr(0, prefix.size()) == prefix) {
                what.remove_prefix(prefix.size());
            }
            usage_error(std::string(what));
        }
        finish();
    }

    std::string help() const { return _app.help(); }

  private:
    void help(const CLI::ParseError &e) {
        std::ostringstream ss;
        _app.exit(e, ss, ss); // prints the selected subcommand's help
        _o.command = Command::Help;
        _o.help_text = ss.str();
    }

    std::vector<CLI::Option *> workload_options(CLI::App *sub) {
        GenSpec &s = _o.spec;
        std::vector<CLI::Option *> w;
        w.push_back(
            sub->add_option("-p,--params", _params, "ML-DSA parameter set")->type_name("44|65|87"));
        w.push_back(sub->add_option("-n,--sigs", s.signatures, "Signatures")
                        ->type_name("N")
                        ->check(quiet(CLI::Range(std::uint64_t{0}, MAX_SIGS))));
        w.push_back(
            sub->add_option("-k,--keys", s.keys,
                            "Distinct keys, signature i uses key i % K (0 = one per signature)")
                ->type_name("K")
                ->check(quiet(CLI::Range(std::uint64_t{0}, MAX_SIGS))));
        w.push_back(sub->add_option("--pass-rate", s.pass_rate, "Fraction that verify, in [0, 1]")
                        ->type_name("R")
                        ->check(quiet(CLI::Range(0.0, 1.0))));
        w.push_back(sub->add_option("--tamper", _tamper,
                                    "How failures are made: message, commitment, z-norm, hint "
                                    "(comma separated) or mixed")
                        ->type_name("MODES"));
        w.push_back(sub->add_option("--msg-len", _msg_len, "Message length in bytes")
                        ->type_name("N|MIN:MAX"));
        w.push_back(
            sub->add_option("--ctx-len", s.ctx_max, "Context length uniform in [0, MAX]")
                ->type_name("MAX")
                ->check(quiet(CLI::Range(std::uint64_t{0}, std::uint64_t{MAX_CONTEXT_BYTES}))));
        w.push_back(sub->add_flag("--deterministic", s.deterministic, "Sign with zero randomness"));
        w.push_back(sub->add_option("--seed", s.seed, "Workload seed")->type_name("S"));
        w.push_back(sub->add_option("--gen-impl", _gen_impl, "Implementation that signs")
                        ->type_name(impl_type())
                        ->check(quiet(CLI::IsMember(impl_choices()))));
        for (CLI::Option *opt : w) {
            opt->group(WORKLOAD)->capture_default_str();
            _workload.push_back(opt);
        }
        return w;
    }

    void threads_option(CLI::App *sub) {
        sub->add_option("-t,--threads", _o.threads, "CPU threads (0 = all)")
            ->type_name("N")
            ->check(quiet(CLI::Range(0u, 4096u)))
            ->capture_default_str();
    }

    void verifier_options(CLI::App *sub) {
        sub->add_option("--backend", _backend, "Verification backend")
            ->type_name("cpu|gpu")
            ->check(quiet(CLI::IsMember({"cpu", "gpu"})))
            ->capture_default_str()
            ->group(VERIFICATION);
        sub->add_option("--cpu-impl", _cpu_impl_name,
                        "CPU implementation (auto: avx2 when it supports the workload)")
            ->type_name(impl_type())
            ->check(quiet(CLI::IsMember(impl_choices())))
            ->capture_default_str()
            ->group(VERIFICATION);
        sub->add_flag("--verdicts", _o.verdicts,
                      "Report why each signature fails (pq-crystals ref stages; slower)")
            ->group(VERIFICATION);
        threads_option(sub);
    }

    void finish() {
        bool found = false;
        for (const auto &[command, sub] : _sub) {
            if (sub->parsed()) {
                _o.command = command;
                found = true;
            }
        }
        if (!found) {
            usage_error("no command given: expected info, generate, verify or kat");
        }
        for (const CLI::Option *w : _workload) {
            _o.workload_options = _o.workload_options || w->count() > 0;
        }

        _o.format = _format == "json" ? Format::Json : Format::Text;
        _o.spec.params = parse_params(_params);
        _o.spec.fail_modes = parse_tampers(_tamper);
        parse_msg_len(_msg_len, _o.spec);
        if (!(_o.spec.pass_rate >= 0.0 && _o.spec.pass_rate <= 1.0)) { // NaN passes Range
            usage_error("--pass-rate: expected a number in [0, 1]");
        }
        _o.cpu_impl = parse_impl("--cpu-impl", _cpu_impl_name);
        const auto gen = parse_impl("--gen-impl", _gen_impl);
        _o.spec.impl = gen ? *gen : fastest_impl(Interface::External);
        _o.spec.threads = _o.threads;

        if (_backend == "gpu") {
            usage_error("--backend: the GPU backend is not implemented yet; use --backend cpu");
        }
        if (_o.verdicts && _o.cpu_impl && *_o.cpu_impl != cpuref::Impl::Ref) {
            usage_error("--verdicts runs the pq-crystals ref stages; it cannot be combined with "
                        "--cpu-impl " +
                        std::string(cpuref::impl_name(*_o.cpu_impl)));
        }
        if (_o.command == Command::Generate ||
            (_o.command == Command::Verify && _o.input.empty())) {
            try {
                validate_spec(_o.spec);
            } catch (const std::invalid_argument &e) {
                usage_error(e.what());
            }
        }
    }

    Options &_o;
    CLI::App _app;
    std::map<Command, CLI::App *> _sub;
    std::vector<CLI::Option *> _workload;

    std::string _format = "text";
    std::string _params = "44";
    std::string _tamper = "message";
    std::string _msg_len = "32";
    std::string _gen_impl = "auto";
    std::string _cpu_impl_name = "auto";
    std::string _backend = "cpu";
};

} // namespace

const char *command_name(Command c) {
    switch (c) {
    case Command::Help:
        return "help";
    case Command::Version:
        return "version";
    case Command::Info:
        return "info";
    case Command::Generate:
        return "generate";
    case Command::Verify:
        return "verify";
    case Command::Kat:
        return "kat";
    }
    return "unknown";
}

cpuref::Impl fastest_impl(Interface iface) {
    return cpuref::supports(cpuref::Impl::Avx2, iface) ? cpuref::Impl::Avx2 : cpuref::Impl::Ref;
}

Options parse_options(int argc, const char *const *argv) {
    Options o;
    Cli cli(o);
    cli.parse(argc, argv);
    return o;
}

std::string usage_text() {
    Options o;
    const Cli cli(o);
    return cli.help();
}

} // namespace ppq::app
