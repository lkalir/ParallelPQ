#include "ppq/verify_kernels.h"

#include <stdexcept>

#include "kernel_util.h"
#include "ppq/decode.h"
#include "ppq/mldsa_traits.h"
#include "ppq/mu.h"
#include "ppq/ntt.h"
#include "ppq/ntt_kernels.h"
#include "ppq/reduce.h"
#include "ppq/rounding.h"
#include "ppq/sample.h"

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

// ---------------------------------------------------------------------- ExpandA and challenge

template <int P> __global__ void expand_a_kernel(KeyInputs keys, KeyCacheView cache) {
    using M = Mldsa<P>;
    constexpr std::size_t entries = static_cast<std::size_t>(M::k) * M::l;
    const std::size_t jobs = keys.count * entries;
    for (std::size_t t = grid_start(); t < jobs; t += grid_stride()) {
        const std::size_t key = t / entries;
        const std::size_t e = t % entries;
        expand_a_poly(keys.pks + key * M::pk_bytes, static_cast<unsigned>(e / M::l),
                      static_cast<unsigned>(e % M::l), cache.a_hat + (key * entries + e) * N);
    }
}

template <int P> __global__ void challenge_kernel(SigInputs in, SigWork work) {
    using M = Mldsa<P>;
    for (std::size_t sig = grid_start(); sig < in.count; sig += grid_stride()) {
        std::int32_t *c = work.c + sig * N;
        sample_in_ball<P>(in.sigs + sig * M::sig_bytes, c);
        ntt(c);
    }
}

template <int P> __global__ void key_t1_kernel(KeyInputs keys, KeyCacheView cache) {
    using M = Mldsa<P>;
    const std::size_t jobs = keys.count * M::k;
    for (std::size_t t = grid_start(); t < jobs; t += grid_stride()) {
        const std::size_t key = t / M::k;
        const std::size_t i = t % M::k;
        std::int32_t *a = cache.t1_hat + (key * M::k + i) * N;
        t1_decode_poly(keys.pks + key * M::pk_bytes + SEED_BYTES + i * T1_PACKED_BYTES, a);
        for (unsigned c = 0; c < N; ++c) {
            a[c] <<= D; // t1 < 2^10, so t1 * 2^d < 2^23 < q: a valid NTT input
        }
        ntt(a);
    }
}

// --------------------------------------------------------------------------------- arithmetic

/// |A_hat| < q and |z_hat| < 9q, so each product is below 9q^2 < 2^31 q; each Montgomery
/// product is in (-q, q) and the l + 1 of them sum to below 8q, well inside reduce32's range.
/// reduce32 leaves |w| < q, the inverse NTT's input bound.
template <int P> __global__ void matvec_kernel(SigInputs in, KeyCacheView cache, SigWork work) {
    using M = Mldsa<P>;
    const std::size_t jobs = in.count * M::k * N;
    for (std::size_t t = grid_start(); t < jobs; t += grid_stride()) {
        const std::size_t sig = t / (M::k * N);
        const std::size_t i = (t / N) % M::k;
        const std::size_t c = t % N;
        const std::size_t key = in.key_index[sig];

        const std::int32_t *a = cache.a_hat + (key * M::k + i) * M::l * N + c;
        const std::int32_t *z = work.z + sig * M::l * N + c;
        std::int32_t acc = 0;
        for (int j = 0; j < M::l; ++j) {
            acc += mont_mul(a[j * N], z[j * N]);
        }
        acc -= mont_mul(work.c[sig * N + c], cache.t1_hat[(key * M::k + i) * N + c]);
        work.w[(sig * M::k + i) * N + c] = reduce32(acc);
    }
}

template <int P> __global__ void use_hint_kernel(SigInputs in, SigWork work) {
    using M = Mldsa<P>;
    const std::size_t jobs = in.count * M::k;
    for (std::size_t t = grid_start(); t < jobs; t += grid_stride()) {
        const std::size_t sig = t / M::k;
        const std::size_t i = t % M::k;
        std::int32_t *w = work.w + (sig * M::k + i) * N;
        const std::uint8_t *h = work.h + (sig * M::k + i) * N;
        // w_approx in [0, q), then w1, reusing w's storage
        for (unsigned c = 0; c < N; ++c) {
            w[c] = use_hint<P>(h[c], caddq(w[c]));
        }
        w1_pack_poly<P>(w, work.w1 + sig * M::w1_bytes + i * M::w1_poly_bytes);
    }
}

// ------------------------------------------------------------------------------ commitment

template <int P> __global__ void final_hash_kernel(SigInputs in, SigWork work) {
    using M = Mldsa<P>;
    for (std::size_t sig = grid_start(); sig < in.count; sig += grid_stride()) {
        Shake256 h;
        h.absorb(work.mu + sig * MU_BYTES, MU_BYTES);
        h.absorb(work.w1 + sig * M::w1_bytes, M::w1_bytes);
        std::uint8_t ctilde[M::ctilde_bytes];
        h.squeeze(ctilde, M::ctilde_bytes);

        const std::uint8_t *want = in.sigs + sig * M::sig_bytes;
        std::uint8_t diff = 0;
        for (std::size_t i = 0; i < M::ctilde_bytes; ++i) {
            diff |= static_cast<std::uint8_t>(ctilde[i] ^ want[i]);
        }
        if (diff != 0) {
            work.status[sig] |= STATUS_COMMITMENT_BAD;
        }
    }
}

/// Queried once per parameter set: kernel attributes do not change while the program runs, and
/// querying them on every launch would cost a dozen API calls per stage
template <int P> int max_threads_for() {
    static const int cached = kernel::min_max_threads(
        &decode_z_kernel<P>, &decode_hint_kernel<P>, &key_tr_kernel<P>, &mu_kernel,
        &expand_a_kernel<P>, &challenge_kernel<P>, &key_t1_kernel<P>, &matvec_kernel<P>,
        &use_hint_kernel<P>, &final_hash_kernel<P>);
    return cached;
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

void launch_key_expand_a(const ParamSet &p, const KeyInputs &keys, const KeyCacheView &cache,
                         unsigned threads, hipStream_t stream) {
    with_params(p.id, [&](auto m) {
        using M = decltype(m);
        constexpr int P = M::id;
        const Launch l =
            launch_for(keys.count * M::k * M::l, threads, max_threads_for<P>(), "expand_a_kernel");
        hipLaunchKernelGGL(expand_a_kernel<P>, kernel::grid_of(l), kernel::block_of(l), 0, stream,
                           keys, cache);
        HIP_CHECK(hipGetLastError());
    });
}

void launch_challenge(const ParamSet &p, const SigInputs &in, const SigWork &work, unsigned threads,
                      hipStream_t stream) {
    with_params(p.id, [&](auto m) {
        constexpr int P = decltype(m)::id;
        const Launch l = launch_for(in.count, threads, max_threads_for<P>(), "challenge_kernel");
        hipLaunchKernelGGL(challenge_kernel<P>, kernel::grid_of(l), kernel::block_of(l), 0, stream,
                           in, work);
        HIP_CHECK(hipGetLastError());
    });
}

void launch_key_t1(const ParamSet &p, const KeyInputs &keys, const KeyCacheView &cache,
                   unsigned threads, hipStream_t stream) {
    with_params(p.id, [&](auto m) {
        using M = decltype(m);
        constexpr int P = M::id;
        const Launch l =
            launch_for(keys.count * M::k, threads, max_threads_for<P>(), "key_t1_kernel");
        hipLaunchKernelGGL(key_t1_kernel<P>, kernel::grid_of(l), kernel::block_of(l), 0, stream,
                           keys, cache);
        HIP_CHECK(hipGetLastError());
    });
}

void launch_prepare_keys(const ParamSet &p, const KeyInputs &keys, const KeyCacheView &cache,
                         unsigned threads, hipStream_t stream) {
    if (keys.count > cache.count) {
        throw std::invalid_argument("the key cache is smaller than the key table");
    }
    launch_key_tr(p, keys, cache, threads, stream);
    launch_key_expand_a(p, keys, cache, threads, stream);
    launch_key_t1(p, keys, cache, threads, stream);
}

void launch_z_ntt(const ParamSet &p, const SigInputs &in, const SigWork &work, unsigned ntt_group,
                  unsigned threads, hipStream_t stream) {
    const std::size_t polys = in.count * static_cast<std::size_t>(p.l);
    launch_ntt_coop(work.z, polys, NttDirection::Forward, ntt_group,
                    kernel::cover(polys * ntt_group, threads), stream);
}

void launch_matvec(const ParamSet &p, const SigInputs &in, const KeyCacheView &cache,
                   const SigWork &work, unsigned threads, hipStream_t stream) {
    with_params(p.id, [&](auto m) {
        using M = decltype(m);
        constexpr int P = M::id;
        const Launch l =
            launch_for(in.count * M::k * N, threads, max_threads_for<P>(), "matvec_kernel");
        hipLaunchKernelGGL(matvec_kernel<P>, kernel::grid_of(l), kernel::block_of(l), 0, stream, in,
                           cache, work);
        HIP_CHECK(hipGetLastError());
    });
}

void launch_w_intt(const ParamSet &p, const SigInputs &in, const SigWork &work, unsigned ntt_group,
                   unsigned threads, hipStream_t stream) {
    const std::size_t polys = in.count * static_cast<std::size_t>(p.k);
    launch_ntt_coop(work.w, polys, NttDirection::Inverse, ntt_group,
                    kernel::cover(polys * ntt_group, threads), stream);
}

void launch_use_hint(const ParamSet &p, const SigInputs &in, const SigWork &work, unsigned threads,
                     hipStream_t stream) {
    with_params(p.id, [&](auto m) {
        using M = decltype(m);
        constexpr int P = M::id;
        const Launch l =
            launch_for(in.count * M::k, threads, max_threads_for<P>(), "use_hint_kernel");
        hipLaunchKernelGGL(use_hint_kernel<P>, kernel::grid_of(l), kernel::block_of(l), 0, stream,
                           in, work);
        HIP_CHECK(hipGetLastError());
    });
}

void launch_final_hash(const ParamSet &p, const SigInputs &in, const SigWork &work,
                       unsigned threads, hipStream_t stream) {
    with_params(p.id, [&](auto m) {
        constexpr int P = decltype(m)::id;
        const Launch l = launch_for(in.count, threads, max_threads_for<P>(), "final_hash_kernel");
        hipLaunchKernelGGL(final_hash_kernel<P>, kernel::grid_of(l), kernel::block_of(l), 0, stream,
                           in, work);
        HIP_CHECK(hipGetLastError());
    });
}

} // namespace ppq
