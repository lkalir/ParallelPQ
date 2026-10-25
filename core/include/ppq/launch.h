#ifndef PPQ_LAUNCH_H_
#define PPQ_LAUNCH_H_

#include <cstddef>
#include <stdexcept>
#include <string>

namespace ppq {

struct Launch {
    std::size_t blocks = 0;
    std::size_t threads = 0;
};

/// A kernel's register use caps its threads per block below the device limit
inline void require_block_size(const Launch &launch, int max_threads, const char *kernel) {
    if (launch.threads == 0 || launch.threads > static_cast<std::size_t>(max_threads)) {
        throw std::invalid_argument(
            std::string(kernel) + " runs 1 to " + std::to_string(max_threads) +
            " threads per block on this device, got " + std::to_string(launch.threads));
    }
}

} // namespace ppq

#endif
