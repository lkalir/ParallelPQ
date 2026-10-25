#include "ppq/hash_kernels.h"

#include <algorithm>

#include "ppq/hip_check.h"

namespace ppq {
namespace {

__device__ __forceinline__ std::size_t grid_start() {
    return static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ std::size_t grid_stride() {
    return static_cast<std::size_t>(gridDim.x) * blockDim.x;
}

__global__ void hash_mu_kernel(BatchLayout layout, BatchView v) {
    for (std::size_t s = grid_start(); s < v.count; s += grid_stride()) {
        hash_mu_one(v.pk + s * layout.pk_stride(), layout.pk_stride(),
                    v.msg + s * layout.msg_stride(), layout.msg_stride(),
                    v.digest + s * layout.digest_stride());
    }
}

__global__ void expand_a_kernel(BatchLayout layout, BatchView v) {
    const std::size_t entries = layout.matrix_entries();
    const std::size_t jobs = v.count * entries;
    const auto l = static_cast<std::size_t>(layout.params.l);

    for (std::size_t t = grid_start(); t < jobs; t += grid_stride()) {
        const std::size_t s = t / entries;
        const std::size_t e = t % entries;
        expand_a_one(v.pk + s * layout.pk_stride(), static_cast<int>(e / l),
                     static_cast<int>(e % l), v.matrix + t * EXPAND_A_BYTES);
    }
}

int max_threads(const void *kernel) {
    hipFuncAttributes attr{};
    HIP_CHECK(hipFuncGetAttributes(&attr, kernel));
    return attr.maxThreadsPerBlock;
}

dim3 grid(const Launch &l) { return dim3(static_cast<unsigned>(l.blocks)); }
dim3 block(const Launch &l) { return dim3(static_cast<unsigned>(l.threads)); }

} // namespace

int hash_kernels_max_threads() {
    return std::min(max_threads(reinterpret_cast<const void *>(&hash_mu_kernel)),
                    max_threads(reinterpret_cast<const void *>(&expand_a_kernel)));
}

void launch_hash_mu(const BatchLayout &layout, const BatchView &v, const Launch &launch,
                    hipStream_t stream) {
    if (v.count == 0) {
        return;
    }
    require_block_size(launch, max_threads(reinterpret_cast<const void *>(&hash_mu_kernel)),
                       "hash_mu_kernel");
    hipLaunchKernelGGL(hash_mu_kernel, grid(launch), block(launch), 0, stream, layout, v);
    HIP_CHECK(hipGetLastError());
}

void launch_expand_a(const BatchLayout &layout, const BatchView &v, const Launch &launch,
                     hipStream_t stream) {
    if (v.count == 0) {
        return;
    }
    require_block_size(launch, max_threads(reinterpret_cast<const void *>(&expand_a_kernel)),
                       "expand_a_kernel");
    hipLaunchKernelGGL(expand_a_kernel, grid(launch), block(launch), 0, stream, layout, v);
    HIP_CHECK(hipGetLastError());
}

} // namespace ppq
