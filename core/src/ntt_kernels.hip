#include "ppq/ntt_kernels.h"

#include <stdexcept>
#include <string>

#include "kernel_util.h"
#include "ppq/ntt.h"
#include "ppq/ntt_block.h"

namespace ppq {
namespace {

using kernel::grid_start;
using kernel::grid_stride;

template <NttDirection Dir>
__global__ void ntt_simple_kernel(std::int32_t *polys, std::size_t count) {
    for (std::size_t p = grid_start(); p < count; p += grid_stride()) {
        std::int32_t *a = polys + p * N;
        if constexpr (Dir == NttDirection::Forward) {
            ntt(a);
        } else {
            invntt_tomont(a);
        }
    }
}

/// Each group of T threads owns one polynomial slot in shared memory; the block strides over
/// the batch `groups` polynomials at a time. The loop bound depends only on blockIdx, so every
/// thread reaches the same barriers; groups past the end of the batch run inactive.
template <unsigned T, NttDirection Dir>
__global__ void ntt_coop_kernel(std::int32_t *polys, std::size_t count) {
    HIP_DYNAMIC_SHARED(std::int32_t, smem)
    const unsigned groups = blockDim.x / T;
    const unsigned group = threadIdx.x / T;
    const unsigned t = threadIdx.x % T;
    std::int32_t *s = smem + group * N;

    for (std::size_t base = static_cast<std::size_t>(blockIdx.x) * groups; base < count;
         base += static_cast<std::size_t>(gridDim.x) * groups) {
        const std::size_t p = base + group;
        const bool active = p < count;
        std::int32_t *g = polys + p * N;

        if (active) {
            for (unsigned c = t; c < N; c += T) {
                s[c] = g[c];
            }
        }
        __syncthreads();

        if constexpr (Dir == NttDirection::Forward) {
            ntt_block<T>(s, t, active);
        } else {
            invntt_tomont_block<T>(s, t, active);
        }

        if (active) {
            for (unsigned c = t; c < N; c += T) {
                g[c] = s[c];
            }
        }
        __syncthreads();
    }
}

template <unsigned T> int coop_max_threads() {
    return kernel::min_max_threads(&ntt_coop_kernel<T, NttDirection::Forward>,
                                   &ntt_coop_kernel<T, NttDirection::Inverse>);
}

template <unsigned T>
void launch_coop(std::int32_t *polys, std::size_t count, NttDirection dir, const Launch &launch,
                 hipStream_t stream) {
    const std::size_t shared = launch.threads / T * N * sizeof(std::int32_t);
    if (dir == NttDirection::Forward) {
        hipLaunchKernelGGL((ntt_coop_kernel<T, NttDirection::Forward>), kernel::grid_of(launch),
                           kernel::block_of(launch), shared, stream, polys, count);
    } else {
        hipLaunchKernelGGL((ntt_coop_kernel<T, NttDirection::Inverse>), kernel::grid_of(launch),
                           kernel::block_of(launch), shared, stream, polys, count);
    }
}

} // namespace

bool valid_ntt_group_size(unsigned group) {
    for (const unsigned g : NTT_GROUP_SIZES) {
        if (g == group) {
            return true;
        }
    }
    return false;
}

int ntt_coop_max_threads(unsigned group) {
    switch (group) {
    case 16:
        return coop_max_threads<16>();
    case 32:
        return coop_max_threads<32>();
    case 64:
        return coop_max_threads<64>();
    case 128:
        return coop_max_threads<128>();
    default:
        throw std::invalid_argument("NTT group size must be 16, 32, 64 or 128, got " +
                                    std::to_string(group));
    }
}

void launch_ntt_coop(std::int32_t *polys, std::size_t count, NttDirection dir, unsigned group,
                     const Launch &launch, hipStream_t stream) {
    require_block_size(launch, ntt_coop_max_threads(group), "ntt_coop_kernel");
    if (launch.threads % group != 0) {
        throw std::invalid_argument(
            "ntt_coop_kernel needs threads per block (" + std::to_string(launch.threads) +
            ") to be a multiple of the group size (" + std::to_string(group) + ")");
    }
    switch (group) {
    case 16:
        launch_coop<16>(polys, count, dir, launch, stream);
        break;
    case 32:
        launch_coop<32>(polys, count, dir, launch, stream);
        break;
    case 64:
        launch_coop<64>(polys, count, dir, launch, stream);
        break;
    case 128:
        launch_coop<128>(polys, count, dir, launch, stream);
        break;
    }
    HIP_CHECK(hipGetLastError());
}

int ntt_simple_max_threads() {
    return kernel::min_max_threads(&ntt_simple_kernel<NttDirection::Forward>,
                                   &ntt_simple_kernel<NttDirection::Inverse>);
}

void launch_ntt_simple(std::int32_t *polys, std::size_t count, NttDirection dir,
                       const Launch &launch, hipStream_t stream) {
    require_block_size(launch, ntt_simple_max_threads(), "ntt_simple_kernel");
    const dim3 grid = kernel::grid_of(launch);
    const dim3 block = kernel::block_of(launch);
    if (dir == NttDirection::Forward) {
        hipLaunchKernelGGL(ntt_simple_kernel<NttDirection::Forward>, grid, block, 0, stream, polys,
                           count);
    } else {
        hipLaunchKernelGGL(ntt_simple_kernel<NttDirection::Inverse>, grid, block, 0, stream, polys,
                           count);
    }
    HIP_CHECK(hipGetLastError());
}

} // namespace ppq
