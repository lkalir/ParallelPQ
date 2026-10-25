#ifndef TESTS_GPU_TEST_H_
#define TESTS_GPU_TEST_H_

#include <hip/hip_runtime.h>

#include <iostream>

#include "check.h"
#include "ppq/launch.h"

namespace gpu_test {

/// CTest treats this exit code as "skipped"
constexpr int SKIP_EXIT_CODE = 77;

inline int &warp_size() {
    static int w = 32;
    return w;
}

/// Threads per block that are a multiple of the warp size on every platform
inline ppq::Launch small_launch() { return {4, 64}; }

inline int run_all() {
    int count = 0;

    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
        std::cout << "no HIP device visible; skipping GPU tests\n";
        return SKIP_EXIT_CODE;
    }

    hipDeviceProp_t props{};
    if (hipSetDevice(0) != hipSuccess || hipGetDeviceProperties(&props, 0) != hipSuccess) {
        std::cerr << "cannot open device 0\n";
        return 1;
    }

    warp_size() = props.warpSize;
    std::cout << "device 0: " << props.name << " (warp size " << props.warpSize << ")\n";
    return check::run_all();
}

} // namespace gpu_test

#endif
