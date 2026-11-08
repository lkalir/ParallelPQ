#ifndef PPQ_APP_GPU_QUERY_H_
#define PPQ_APP_GPU_QUERY_H_

/// What `ppq info` reports about GPUs. Implemented by gpu_query.hip in GPU builds; CPU-only
/// builds pass no query function.

#include <string>
#include <vector>

#include "ppq/device.h"

namespace ppq::app {

struct GpuQuery {
    /// "nvidia" or "amd": the platform this build targets
    std::string platform;
    std::vector<DeviceInfo> devices;
    /// Set when the runtime could not be queried
    std::string error;
};

using GpuQueryFn = GpuQuery (*)();

/// Defined only in GPU builds
GpuQuery query_gpus();

} // namespace ppq::app

#endif
