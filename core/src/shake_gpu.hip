#include "ppq/shake_gpu.h"

#include <algorithm>

#include "ppq/buffer.h"
#include "ppq/hip_check.h"
#include "ppq/keccak.h"

namespace ppq {
namespace {

constexpr std::size_t INPUT_ALIGN = 8;

struct ShakeJob {
    std::uint64_t in_offset;
    std::uint64_t in_len;
    std::uint64_t out_offset;
    std::uint64_t out_len;
    ShakeVariant variant;
};

__global__ void shake_jobs_kernel(const ShakeJob *jobs, std::size_t count, const std::uint8_t *in,
                                  std::uint8_t *out) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;

    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += stride) {
        const ShakeJob job = jobs[i];

        if (job.variant == ShakeVariant::Shake128) {
            shake<Shake128>(in + job.in_offset, job.in_len, out + job.out_offset, job.out_len);
        } else {
            shake<Shake256>(in + job.in_offset, job.in_len, out + job.out_offset, job.out_len);
        }
    }
}

} // namespace

int shake_gpu_max_threads() {
    hipFuncAttributes attr{};
    HIP_CHECK(hipFuncGetAttributes(&attr, reinterpret_cast<const void *>(&shake_jobs_kernel)));
    return attr.maxThreadsPerBlock;
}

std::vector<std::vector<std::uint8_t>> shake_gpu(const std::vector<ShakeRequest> &requests,
                                                 const Launch &launch, std::size_t misalign,
                                                 hipStream_t stream) {
    require_block_size(launch, shake_gpu_max_threads(), "shake_jobs_kernel");

    std::vector<ShakeJob> jobs(requests.size());
    std::size_t in_bytes = 0;
    std::size_t out_bytes = 0;

    for (std::size_t i = 0; i < requests.size(); ++i) {
        const std::size_t aligned = (in_bytes + INPUT_ALIGN - 1) / INPUT_ALIGN * INPUT_ALIGN;
        jobs[i] = {aligned + misalign, requests[i].in_len, out_bytes, requests[i].out_len,
                   requests[i].variant};
        in_bytes = aligned + misalign + requests[i].in_len;
        out_bytes += requests[i].out_len;
    }

    std::vector<std::uint8_t> packed(in_bytes + 1, 0);
    for (std::size_t i = 0; i < requests.size(); ++i) {
        std::copy(requests[i].input, requests[i].input + requests[i].in_len,
                  packed.begin() + static_cast<std::ptrdiff_t>(jobs[i].in_offset));
    }

    DeviceBuffer<ShakeJob> d_jobs(jobs);
    DeviceBuffer<std::uint8_t> d_in(packed);
    DeviceBuffer<std::uint8_t> d_out(out_bytes + 1);

    hipLaunchKernelGGL(shake_jobs_kernel, dim3(static_cast<unsigned>(launch.blocks)),
                       dim3(static_cast<unsigned>(launch.threads)), 0, stream, d_jobs.get(),
                       jobs.size(), d_in.get(), d_out.get());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<std::uint8_t> flat(out_bytes + 1);
    d_out.download(flat);

    std::vector<std::vector<std::uint8_t>> results(requests.size());
    for (std::size_t i = 0; i < requests.size(); ++i) {
        const auto first = flat.begin() + static_cast<std::ptrdiff_t>(jobs[i].out_offset);
        results[i].assign(first, first + static_cast<std::ptrdiff_t>(jobs[i].out_len));
    }

    return results;
}

} // namespace ppq
