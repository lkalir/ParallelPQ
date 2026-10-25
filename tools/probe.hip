/// Tester for HIP

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ppq/stream.h"

namespace {

#define PROBE_CHECK(call)                                                                          \
    do {                                                                                           \
        const hipError_t err_ = (call);                                                            \
        if (err_ != hipSuccess) {                                                                  \
            std::fprintf(stderr, "%s:%d: %s failed: %s (%s)\n", __FILE__, __LINE__, #call,         \
                         hipGetErrorName(err_), hipGetErrorString(err_));                          \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

constexpr int COEFFS = 4;
constexpr int BLOCK = 256;

__constant__ float c_coeffs[COEFFS];

__global__ void touch_kernel(int *flag) { *flag += 1; }

__global__ void probe_kernel(const float *in, float *out, int n) {
    __shared__ float tile[BLOCK];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float mine = (i < n) ? in[i] : 0.0f; // register

    tile[threadIdx.x] = mine;
    __syncthreads();

    const float mirror = tile[blockDim.x - 1 - threadIdx.x];
    const float coeff = c_coeffs[threadIdx.x % COEFFS];

    if (i < n) {
        out[i] = coeff * mine + mirror;
    }
}

const char *yes_no(int v) { return v ? "yes" : "no"; }

void print_device(int dev, const hipDeviceProp_t &p) {
    std::printf("device %d: %s\n", dev, p.name);
#if defined(__HIP_PLATFORM_AMD__)
    std::printf("  arch                     : %s\n", p.gcnArchName);
#else
    std::printf("  arch                     : sm_%d%d\n", p.major, p.minor);
#endif
    std::printf("  compute units / SMs      : %d\n", p.multiProcessorCount);
    std::printf("  warp size                : %d\n", p.warpSize);
    std::printf("  max threads per block    : %d\n", p.maxThreadsPerBlock);
    std::printf("  global memory            : %.2f GiB\n",
                static_cast<double>(p.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
    std::printf("  constant memory          : %zu KiB\n", p.totalConstMem / 1024);
    std::printf("  shared memory per block  : %zu KiB (opt-in max %zu KiB)\n",
                p.sharedMemPerBlock / 1024, p.sharedMemPerBlockOptin / 1024);
    std::printf("  registers per block      : %d\n", p.regsPerBlock);
    std::printf("  L2 cache                 : %d KiB\n", p.l2CacheSize / 1024);
    std::printf("  memory bus width         : %d bits\n", p.memoryBusWidth);
    std::printf("  integrated (shares RAM)  : %s\n", yes_no(p.integrated));
    std::printf("  can map host memory      : %s\n", yes_no(p.canMapHostMemory));
    std::printf("  managed memory           : %s\n", yes_no(p.managedMemory));
}

bool run_kernel(int dev) {
    PROBE_CHECK(hipSetDevice(dev));

    // deliberately not a multiple of the block
    constexpr int n = 4 * BLOCK + 37;
    const float coeffs[COEFFS] = {1.0f, 2.0f, 0.5f, -1.0f};

    std::vector<float> host_in(n);

    for (int i = 0; i < n; ++i) {
        host_in[i] = static_cast<float>(i % 97) * 0.25f;
    }

    float *d_in = nullptr;
    float *d_out = nullptr;

    PROBE_CHECK(hipMalloc(&d_in, n * sizeof(float)));
    PROBE_CHECK(hipMalloc(&d_out, n * sizeof(float)));
    PROBE_CHECK(hipMemcpy(d_in, host_in.data(), n * sizeof(float), hipMemcpyHostToDevice));
    PROBE_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(c_coeffs), coeffs, sizeof(coeffs)));

    const int blocks = (n + BLOCK - 1) / BLOCK;
    hipLaunchKernelGGL(probe_kernel, dim3(blocks), dim3(BLOCK), 0, nullptr, d_in, d_out, n);
    PROBE_CHECK(hipGetLastError());
    PROBE_CHECK(hipDeviceSynchronize());

    std::vector<float> host_out(n);
    PROBE_CHECK(hipMemcpy(host_out.data(), d_out, n * sizeof(float), hipMemcpyDeviceToHost));
    PROBE_CHECK(hipFree(d_in));
    PROBE_CHECK(hipFree(d_out));

    int bad = 0;

    for (int i = 0; i < n; ++i) {
        const int block_start = (i / BLOCK) * BLOCK;
        const int t = i - block_start;
        const int j = block_start + (BLOCK - 1 - t);
        const float mirror = (j < n) ? host_in[j] : 0.0f;
        const float expect = coeffs[t % COEFFS] * host_in[i] + mirror;

        if (host_out[i] != expect) {
            if (bad < 5) {
                std::printf("  mismatch at %d: got %g, expected %g\n", i, host_out[i], expect);
            }
            ++bad;
        }
    }

    std::printf("  kernel: %s\n", bad == 0 ? "PASS" : "FAIL");
    return bad == 0;
}

bool run_streams(int dev) {
    PROBE_CHECK(hipSetDevice(dev));

    int *d_flag = nullptr;
    PROBE_CHECK(hipMalloc(&d_flag, sizeof(int)));
    PROBE_CHECK(hipMemset(d_flag, 0, sizeof(int)));

    ppq::Stream a;
    ppq::Stream b;
    ppq::Event start;
    ppq::Event a_done;
    ppq::Event stop;

    start.record(a.get());
    hipLaunchKernelGGL(touch_kernel, dim3(1), dim3(1), 0, a.get(), d_flag);
    a_done.record(a.get());
    b.wait(a_done);
    hipLaunchKernelGGL(touch_kernel, dim3(1), dim3(1), 0, b.get(), d_flag);
    stop.record(b.get());
    PROBE_CHECK(hipGetLastError());
    stop.synchronize();

    int flag = 0;
    PROBE_CHECK(hipMemcpy(&flag, d_flag, sizeof(int), hipMemcpyDeviceToHost));
    PROBE_CHECK(hipFree(d_flag));

    const float ms = ppq::elapsed_ms(start, stop);
    const bool ok = flag == 2 && a_done.done() && ms >= 0.0f;
    std::printf("  streams/events: %s (%.3f ms across two streams)\n", ok ? "PASS" : "FAIL", ms);
    return ok;
}

} // namespace

int main() {
    int runtime = 0;
    int driver = 0;
    PROBE_CHECK(hipRuntimeGetVersion(&runtime));
    PROBE_CHECK(hipDriverGetVersion(&driver));

#if defined(__HIP_PLATFORM_AMD__)
    const char *platform = "amd";
#else
    const char *platform = "nvidia";
#endif

    std::printf("HIP platform  : %s\n", platform);
    std::printf("HIP headers   : %d.%d.%d\n", HIP_VERSION_MAJOR, HIP_VERSION_MINOR,
                HIP_VERSION_PATCH);
    std::printf("runtime/driver: %d / %d\n", runtime, driver);

    int count = 0;
    PROBE_CHECK(hipGetDeviceCount(&count));
    std::printf("devices       : %d\n\n", count);

    if (count == 0) {
        std::fprintf(stderr, "no HIP devices visible\n");
        return 1;
    }

    bool ok = true;

    for (int dev = 0; dev < count; ++dev) {
        hipDeviceProp_t props{};
        PROBE_CHECK(hipGetDeviceProperties(&props, dev));
        print_device(dev, props);
        ok &= run_kernel(dev);
        ok &= run_streams(dev);
        std::printf("\n");
    }

    std::printf("%s\n", ok ? "probe: all devices passed" : "probe: FAILED");
    return ok ? 0 : 1;
}
