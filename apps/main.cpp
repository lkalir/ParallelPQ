/// ppq.exe: batched ML-DSA verification

#include <exception>
#include <iostream>

#include "commands.h"
#include "options.h"

int main(int argc, char **argv) {
    ppq::app::Options options;
    try {
        options = ppq::app::parse_options(argc, argv);
    } catch (const ppq::app::UsageError &e) {
        std::cerr << "ppq: " << e.what() << "\nrun 'ppq.exe --help' for usage\n";
        return 2;
    }

#if PPQ_WITH_GPU
    ppq::app::Context ctx{std::cout, std::cerr, &ppq::app::query_gpus};
#else
    ppq::app::Context ctx{std::cout, std::cerr, nullptr};
#endif

    try {
        return ppq::app::run_command(options, ctx);
    } catch (const std::exception &e) {
        std::cerr << "ppq: error: " << e.what() << "\n";
        return 1;
    }
}
