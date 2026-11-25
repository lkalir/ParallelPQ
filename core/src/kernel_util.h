#ifndef PPQ_SRC_KERNEL_UTIL_H_
#define PPQ_SRC_KERNEL_UTIL_H_

/// Helpers shared by the kernel sources (private to core/src)

#include <hip/hip_runtime.h>

#include <cstddef>

#include "ppq/hip_check.h"
#include "ppq/launch.h"

namespace ppq::kernel {

__device__ inline std::size_t grid_start() {
    return static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

__device__ inline std::size_t grid_stride() {
    return static_cast<std::size_t>(gridDim.x) * blockDim.x;
}

/// Callers cache the result: attributes are fixed for the program's lifetime (on the device
/// that was current at the first query)
inline int max_threads(const void *kernel) {
    hipFuncAttributes attr{};
    HIP_CHECK(hipFuncGetAttributes(&attr, kernel));
    return attr.maxThreadsPerBlock;
}

template <class... Kernels> int min_max_threads(Kernels... kernels) {
    int best = 1 << 30;
    for (const int t : {max_threads(reinterpret_cast<const void *>(kernels))...}) {
        best = t < best ? t : best;
    }
    return best;
}

inline dim3 grid_of(const Launch &l) { return dim3(static_cast<unsigned>(l.blocks)); }
inline dim3 block_of(const Launch &l) { return dim3(static_cast<unsigned>(l.threads)); }

/// Enough blocks of `threads` to give every one of `jobs` its own thread, capped at `max_blocks`
inline Launch cover(std::size_t jobs, std::size_t threads, std::size_t max_blocks = 65535) {
    std::size_t blocks = (jobs + threads - 1) / threads;
    blocks = blocks == 0 ? 1 : blocks;
    return {blocks < max_blocks ? blocks : max_blocks, threads};
}

} // namespace ppq::kernel

#endif
