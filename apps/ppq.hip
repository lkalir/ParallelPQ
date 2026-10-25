/// ParallelPQ command-line entry point.
///
/// Phase 0 only provides `info`; generate, verify and kat arrive with the verifier.

#include <hip/hip_runtime.h>

#include <cerrno>
#include <cstdlib>
#include <exception>
#include <getopt.h>
#include <iostream>
#include <ostream>
#include <string>

#include "ppq/device.h"

#ifndef PPQ_VERSION
#define PPQ_VERSION "unknown"
#endif

namespace {

constexpr const char *SHORT_OPTS = "d:hV";

const option LONG_OPTS[] = {
    {"device", required_argument, nullptr, 'd'},
    {"help", no_argument, nullptr, 'h'},
    {"version", no_argument, nullptr, 'V'},
    {nullptr, 0, nullptr, 0},
};

void print_usage(const char *prog, std::ostream &os) {
    os << "ParallelPQ " << PPQ_VERSION << ": batched GPU ML-DSA verification\n"
       << "\n"
       << "usage: " << prog << " [options] <command>\n"
       << "\n"
       << "commands:\n"
       << "  info              list visible GPUs and describe the selected one\n"
       << "\n"
       << "options:\n"
       << "  -d, --device N    GPU to use (default 0)\n"
       << "  -h, --help        show this help\n"
       << "  -V, --version     print the version\n"
       << "\n"
       << "exit status: 0 success, 1 runtime failure, 2 usage error\n";
}

bool parse_device(const char *s, int &out) {
    errno = 0;
    char *end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < 0 || v > 1024) {
        return false;
    }
    out = static_cast<int>(v);
    return true;
}

int cmd_info(int device) {
    const int count = ppq::device_count();
    std::cout << "platform        : " << ppq::platform_name() << " (build)\n"
              << "visible devices : " << count << "\n";

    if (count == 0) {
        std::cerr << "error: no HIP device visible\n";
        return 1;
    }

    ppq::print_device(ppq::open_device(device), std::cout);
    return 0;
}

} // namespace

int main(int argc, char **argv) try {
    int device = 0;
    int opt = 0;

    while ((opt = getopt_long(argc, argv, SHORT_OPTS, LONG_OPTS, nullptr)) != -1) {
        switch (opt) {
        case 'd':
            if (!parse_device(optarg, device)) {
                std::cerr << "error: --device '" << optarg << "' is not a device number\n";
                return 2;
            }
            break;
        case 'h':
            print_usage(argv[0], std::cout);
            return 0;
        case 'V':
            std::cout << "ppq " << PPQ_VERSION << " (" << ppq::platform_name() << ")\n";
            return 0;
        default:
            print_usage(argv[0], std::cerr);
            return 2;
        }
    }

    if (optind >= argc) {
        std::cerr << "error: no command given\n\n";
        print_usage(argv[0], std::cerr);
        return 2;
    }

    const std::string command = argv[optind];

    if (optind + 1 < argc) {
        std::cerr << "error: unexpected argument '" << argv[optind + 1] << "'\n";
        return 2;
    }

    if (command == "info") {
        return cmd_info(device);
    }

    std::cerr << "error: unknown command '" << command << "'\n\n";
    print_usage(argv[0], std::cerr);
    return 2;
} catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
}
