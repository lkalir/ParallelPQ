#ifndef PPQ_DEVICE_H_
#define PPQ_DEVICE_H_

#include <cstddef>
#include <iosfwd>
#include <string>

namespace ppq {

struct DeviceInfo {
    int id = 0;
    /// "nvidia" or "amd"
    std::string platform;
    std::string name;
    std::string arch;
    int warp_size = 0;
    int compute_units = 0;
    int max_threads_per_block = 0;
    int regs_per_block = 0;
    int l2_bytes = 0;
    int memory_bus_bits = 0;
    std::size_t global_bytes = 0;
    std::size_t constant_bytes = 0;
    std::size_t shared_per_block = 0;
    std::size_t shared_per_block_optin = 0;
    /// Unified memory
    bool integrated = false;
    bool can_map_host = false;
    bool managed_memory = false;
};

/// Compile-time platform of this build
const char *platform_name();

int device_count();

/// Selects the device and returns its properties
DeviceInfo open_device(int id);

void print_device(const DeviceInfo &info, std::ostream &os);

} // namespace ppq

#endif
