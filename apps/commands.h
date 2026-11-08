#ifndef PPQ_APP_COMMANDS_H_
#define PPQ_APP_COMMANDS_H_

/// The ppq.exe commands: info, generate, verify and kat

#include <iosfwd>

#include "gpu_query.h"
#include "options.h"

namespace ppq::app {

struct Context {
    std::ostream &out;
    std::ostream &err;
    /// Set in GPU builds; info reports "CPU only" without it
    GpuQueryFn query_gpus = nullptr;
};

/// Returns the exit status: 0 success, 1 failure (including results that disagree with the
/// expected verdicts). Throws std::exception on runtime errors.
int run_command(const Options &options, Context &ctx);

} // namespace ppq::app

#endif
