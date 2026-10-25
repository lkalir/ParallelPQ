#ifndef PPQ_SHAKE_GPU_H_
#define PPQ_SHAKE_GPU_H_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ppq/launch.h"
#include "ppq/shake_vectors.h"

namespace ppq {

struct ShakeRequest {
    ShakeVariant variant = ShakeVariant::Shake128;
    const std::uint8_t *input = nullptr;
    std::size_t in_len = 0;
    std::size_t out_len = 0;
};

/// Most threads per block shake_gpu can launch on the current device
int shake_gpu_max_threads();

/// One GPU thread per request. Each input is placed `misalign` bytes past an 8-byte boundary.
std::vector<std::vector<std::uint8_t>> shake_gpu(const std::vector<ShakeRequest> &requests,
                                                 const Launch &launch, std::size_t misalign = 0,
                                                 hipStream_t stream = nullptr);

} // namespace ppq

#endif
