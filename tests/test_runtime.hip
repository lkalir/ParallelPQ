/// The RAII wrappers and device helpers every later kernel builds on

#include <hip/hip_runtime.h>

#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "gpu_test.h"
#include "ppq/buffer.h"
#include "ppq/device.h"
#include "ppq/host_buffer.h"
#include "ppq/stream.h"

namespace {

/// Deliberately not a multiple of any block size
constexpr std::size_t RAGGED_COUNT = 4 * 64 + 37;
constexpr std::size_t ROUND_TRIP_COUNT = 1000;

__global__ void affine_kernel(std::uint32_t *out, std::size_t n) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        out[i] = static_cast<std::uint32_t>(3 * i + 1);
    }
}

__global__ void store_kernel(int *flag, int value) { *flag = value; }

__global__ void increment_kernel(int *flag) { *flag += 1; }

void launch_affine(std::uint32_t *out, std::size_t n, hipStream_t stream) {
    const ppq::Launch l = gpu_test::small_launch();
    hipLaunchKernelGGL(affine_kernel, dim3(static_cast<unsigned>(l.blocks)),
                       dim3(static_cast<unsigned>(l.threads)), 0, stream, out, n);
    HIP_CHECK(hipGetLastError());
}

TEST_CASE(device_buffer_round_trip) {
    std::vector<std::uint32_t> host(ROUND_TRIP_COUNT);
    std::iota(host.begin(), host.end(), 7u);

    ppq::DeviceBuffer<std::uint32_t> dev(host);
    CHECK(dev.size() == host.size());
    CHECK(dev.bytes() == host.size() * sizeof(std::uint32_t));

    std::vector<std::uint32_t> back(host.size(), 0);
    dev.download(back);
    CHECK(back == host);
}

TEST_CASE(moved_from_buffer_is_empty) {
    ppq::DeviceBuffer<int> a(16);
    int *const ptr = a.get();
    ppq::DeviceBuffer<int> b(std::move(a));

    CHECK(b.get() == ptr);
    CHECK(b.size() == 16);
    CHECK(a.get() == nullptr);
    CHECK(a.size() == 0);
    CHECK(ppq::DeviceBuffer<int>(0).get() == nullptr);
}

TEST_CASE(grid_stride_kernel_covers_a_ragged_range) {
    ppq::DeviceBuffer<std::uint32_t> dev(RAGGED_COUNT);
    launch_affine(dev.get(), RAGGED_COUNT, nullptr);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<std::uint32_t> got(RAGGED_COUNT);
    dev.download(got);

    std::size_t wrong = 0;
    for (std::size_t i = 0; i < RAGGED_COUNT; ++i) {
        wrong += got[i] != 3 * i + 1;
    }
    CHECK(wrong == 0);
}

TEST_CASE(pinned_async_copies_and_events_on_a_stream) {
    ppq::PinnedBuffer<std::uint32_t> host(RAGGED_COUNT);
    ppq::DeviceBuffer<std::uint32_t> dev(RAGGED_COUNT);
    ppq::Stream stream;
    ppq::Event start;
    ppq::Event stop;

    for (std::size_t i = 0; i < RAGGED_COUNT; ++i) {
        host.get()[i] = 0;
    }

    start.record(stream.get());
    launch_affine(dev.get(), RAGGED_COUNT, stream.get());
    HIP_CHECK(
        hipMemcpyAsync(host.get(), dev.get(), dev.bytes(), hipMemcpyDeviceToHost, stream.get()));
    stop.record(stream.get());
    stop.synchronize();

    CHECK(stop.done());
    CHECK(ppq::elapsed_ms(start, stop) >= 0.0f);
    CHECK(host.get()[RAGGED_COUNT - 1] == 3 * (RAGGED_COUNT - 1) + 1);
}

TEST_CASE(stream_wait_orders_work_across_streams) {
    ppq::DeviceBuffer<int> flag(1);
    ppq::Stream a;
    ppq::Stream b;
    ppq::Event a_done(hipEventDisableTiming);

    hipLaunchKernelGGL(store_kernel, dim3(1), dim3(1), 0, a.get(), flag.get(), 41);
    HIP_CHECK(hipGetLastError());
    a_done.record(a.get());

    b.wait(a_done);
    hipLaunchKernelGGL(increment_kernel, dim3(1), dim3(1), 0, b.get(), flag.get());
    HIP_CHECK(hipGetLastError());
    b.synchronize();

    std::vector<int> got(1);
    flag.download(got);
    CHECK(got[0] == 42);
}

TEST_CASE(hip_check_throws_with_the_error_code) {
    bool threw = false;
    try {
        HIP_CHECK(hipSetDevice(-1));
    } catch (const ppq::HipError &e) {
        threw = e.code() != hipSuccess;
    }
    CHECK(threw);
    HIP_CHECK(hipSetDevice(0));
}

TEST_CASE(open_device_reports_properties_and_rejects_bad_ids) {
    const ppq::DeviceInfo info = ppq::open_device(0);
    CHECK(!info.name.empty());
    CHECK(info.warp_size == gpu_test::warp_size());
    CHECK(info.max_threads_per_block > 0);
    CHECK(info.platform == ppq::platform_name());

    bool threw = false;
    try {
        ppq::open_device(ppq::device_count());
    } catch (const std::out_of_range &) {
        threw = true;
    }
    CHECK(threw);
}

} // namespace

int main() { return gpu_test::run_all(); }
