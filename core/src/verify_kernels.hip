#include "ppq/verify_kernels.h"

#include "kernel_util.h"
#include "ppq/decode.h"
#include "ppq/mldsa_traits.h"
#include "ppq/mu.h"

namespace ppq {
namespace {

using kernel::grid_start;
using kernel::grid_stride;

// ------------------------------------------------------------------------------------- decode

template <int P> __global__ void decode_z_kernel(SigInputs in, SigWork work) {
    using M = Mldsa<P>;
    const std::size_t jobs = in.count * M::l;
    for (std::size_t t = grid_start(); t < jobs; t += grid_stride()) {
        const std::size_t sig = t / M::l;
        const std::size_t j = t % M::l;
        const std::uint8_t *packed =
            in.sigs + sig * M::sig_bytes + M::ctilde_bytes + j * M::z_poly_bytes;
        std::int32_t *z = work.z + (sig * M::l + j) * N;
        z_decode_poly<P>(packed, z);
        if (!z_poly_norm_ok<P>(z)) {
            atomicOr(&work.status[sig], static_cast<std::uint32_t>(STATUS_ZNORM_BAD));
        }
    }
}

template <int P> __global__ void decode_hint_kernel(SigInputs in, SigWork work) {
    using M = Mldsa<P>;
    for (std::size_t sig = grid_start(); sig < in.count; sig += grid_stride()) {
        const std::uint8_t *y = in.sigs + sig * M::sig_bytes + M::sig_bytes - M::hint_bytes;
        if (!hint_decode<P>(y, work.h + sig * M::k * N)) {
            atomicOr(&work.status[sig], static_cast<std::uint32_t>(STATUS_HINT_BAD));
        }
    }
}

// ---------------------------------------------------------------------------------- tr and mu

template <int P> __global__ void key_tr_kernel(KeyInputs keys, KeyCacheView cache) {
    using M = Mldsa<P>;
    for (std::size_t key = grid_start(); key < keys.count; key += grid_stride()) {
        compute_tr(keys.pks + key * M::pk_bytes, M::pk_bytes, cache.tr + key * TR_BYTES);
    }
}

__global__ void mu_kernel(SigInputs in, KeyCacheView cache, SigWork work) {
    for (std::size_t sig = grid_start(); sig < in.count; sig += grid_stride()) {
        const std::uint64_t m0 = in.msg_offsets[sig];
        const std::uint64_t c0 = in.ctx_offsets[sig];
        compute_mu(in.iface, cache.tr + static_cast<std::size_t>(in.key_index[sig]) * TR_BYTES,
                   in.msg_data + m0, in.msg_offsets[sig + 1] - m0, in.ctx_data + c0,
                   in.ctx_offsets[sig + 1] - c0, work.mu + sig * MU_BYTES);
    }
}

template <int P> int max_threads_for() {
    return kernel::min_max_threads(&decode_z_kernel<P>, &decode_hint_kernel<P>, &key_tr_kernel<P>,
                                   &mu_kernel);
}

Launch launch_for(std::size_t jobs, unsigned threads, int max_threads, const char *name) {
    const Launch l = kernel::cover(jobs, threads);
    require_block_size(l, max_threads, name);
    return l;
}

} // namespace

int verify_kernels_max_threads() {
    const int a = max_threads_for<44>();
    const int b = max_threads_for<65>();
    const int c = max_threads_for<87>();
    return a < b ? (a < c ? a : c) : (b < c ? b : c);
}

void launch_decode(const ParamSet &p, const SigInputs &in, const SigWork &work, unsigned threads,
                   hipStream_t stream) {
    with_params(p.id, [&](auto m) {
        using M = decltype(m);
        constexpr int P = M::id;
        const int max_threads = max_threads_for<P>();
        if (in.count > 0) {
            HIP_CHECK(hipMemsetAsync(work.status, 0, in.count * sizeof(std::uint32_t), stream));
        }
        const Launch lz = launch_for(in.count * M::l, threads, max_threads, "decode_z_kernel");
        hipLaunchKernelGGL(decode_z_kernel<P>, kernel::grid_of(lz), kernel::block_of(lz), 0, stream,
                           in, work);
        HIP_CHECK(hipGetLastError());
        const Launch lh = launch_for(in.count, threads, max_threads, "decode_hint_kernel");
        hipLaunchKernelGGL(decode_hint_kernel<P>, kernel::grid_of(lh), kernel::block_of(lh), 0,
                           stream, in, work);
        HIP_CHECK(hipGetLastError());
    });
}

void launch_key_tr(const ParamSet &p, const KeyInputs &keys, const KeyCacheView &cache,
                   unsigned threads, hipStream_t stream) {
    with_params(p.id, [&](auto m) {
        constexpr int P = decltype(m)::id;
        const Launch l = launch_for(keys.count, threads, max_threads_for<P>(), "key_tr_kernel");
        hipLaunchKernelGGL(key_tr_kernel<P>, kernel::grid_of(l), kernel::block_of(l), 0, stream,
                           keys, cache);
        HIP_CHECK(hipGetLastError());
    });
}

void launch_mu(const ParamSet &p, const SigInputs &in, const KeyCacheView &cache,
               const SigWork &work, unsigned threads, hipStream_t stream) {
    with_params(p.id, [&](auto m) {
        constexpr int P = decltype(m)::id;
        const Launch l = launch_for(in.count, threads, max_threads_for<P>(), "mu_kernel");
        hipLaunchKernelGGL(mu_kernel, kernel::grid_of(l), kernel::block_of(l), 0, stream, in, cache,
                           work);
        HIP_CHECK(hipGetLastError());
    });
}

} // namespace ppq
