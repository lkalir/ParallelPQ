#include "ppq/device.h"
#include "ppq/hip_check.h"

#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string>

namespace ppq {

const char *platform_name() {
#if defined(__HIP_PLATFORM_AMD__)
    return "amd";
#else
    return "nvidia";
#endif
}

int device_count() {
    int n = 0;
    HIP_CHECK(hipGetDeviceCount(&n));
    return n;
}

DeviceInfo open_device(int id) {
    const int n = device_count();

    if (id < 0 || id >= n) {
        throw std::out_of_range("device " + std::to_string(id) + " does not exist (" +
                                std::to_string(n) + " visible)");
    }

    HIP_CHECK(hipSetDevice(id));

    hipDeviceProp_t p{};
    HIP_CHECK(hipGetDeviceProperties(&p, id));

    DeviceInfo d;
    d.id = id;
    d.platform = platform_name();
    d.name = p.name;
#if defined(__HIP_PLATFORM_AMD__)
    // gcnArchName carries feature flags after the target
    d.arch = p.gcnArchName;
    d.arch = d.arch.substr(0, d.arch.find(':'));
#else
    d.arch = "sm_" + std::to_string(p.major) + std::to_string(p.minor);
#endif
    d.warp_size = p.warpSize;
    d.compute_units = p.multiProcessorCount;
    d.max_threads_per_block = p.maxThreadsPerBlock;
    d.regs_per_block = p.regsPerBlock;
    d.l2_bytes = p.l2CacheSize;
    d.memory_bus_bits = p.memoryBusWidth;
    d.global_bytes = p.totalGlobalMem;
    d.constant_bytes = p.totalConstMem;
    d.shared_per_block = p.sharedMemPerBlock;
    d.shared_per_block_optin = p.sharedMemPerBlockOptin;
    d.integrated = p.integrated != 0;
    d.can_map_host = p.canMapHostMemory != 0;
    d.managed_memory = p.managedMemory != 0;
    return d;
}

void print_device(const DeviceInfo &d, std::ostream &os) {
    const auto kib = [](std::size_t b) { return b / 1024; };
    const double gib = static_cast<double>(d.global_bytes) / (1024.0 * 1024.0 * 1024.0);
    const std::ios_base::fmtflags flags = os.flags();

    os << "device          : " << d.id << " " << d.name << " (" << d.arch << ", " << d.platform
       << ")\n"
       << "compute units   : " << d.compute_units << ", warp size " << d.warp_size << "\n"
       << "global memory   : " << std::fixed << std::setprecision(1) << gib << " GiB"
       << (d.integrated ? " (integrated: shared with the CPU)" : "") << "\n"
       << "constant memory : " << kib(d.constant_bytes) << " KiB\n"
       << "shared memory   : " << kib(d.shared_per_block) << " KiB per block ("
       << kib(d.shared_per_block_optin) << " KiB opt-in)\n"
       << "registers       : " << d.regs_per_block << " per block\n";

    os.flags(flags);
}

} // namespace ppq
