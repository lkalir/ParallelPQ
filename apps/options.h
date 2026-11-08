#ifndef PPQ_APP_OPTIONS_H_
#define PPQ_APP_OPTIONS_H_

/// Command-line options for ppq.exe

#include <optional>
#include <stdexcept>
#include <string>

#include "ppq/cpuref.h"
#include "ppq/generate.h"

namespace ppq::app {

/// A bad command line: reported with a hint to run --help, exit status 2
class UsageError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

enum class Command {
    Help,
    Version,
    Info,
    Generate,
    Verify,
    Kat,
};

enum class Format {
    Text,
    Json,
};

struct Options {
    Command command = Command::Help;

    /// Workload to generate (generate, and verify without --input). spec.impl is resolved
    /// from --gen-impl; spec.threads from --threads.
    GenSpec spec;
    /// True when any workload option was given explicitly
    bool workload_options = false;

    std::string input;
    std::string output;

    /// Verification implementation; nullopt picks the fastest one for the workload
    std::optional<cpuref::Impl> cpu_impl;
    /// 0 = all hardware threads
    unsigned threads = 0;
    bool verdicts = false;

    unsigned repeat = 1;
    unsigned warmup = 0;

    /// Root of the vector corpus (holds acvp/mldsa_sigver.txt)
    std::string vectors_dir;

    Format format = Format::Text;

    /// For Command::Help: the help asked for (the subcommand's after `<command> --help`)
    std::string help_text;
    /// For Command::Version
    std::string version_text;
};

/// Throws UsageError
Options parse_options(int argc, const char *const *argv);

/// The top-level --help text
std::string usage_text();

/// The fastest available implementation that supports the interface: AVX2, else ref
cpuref::Impl fastest_impl(Interface iface);

const char *command_name(Command c);

} // namespace ppq::app

#endif
