#ifndef PPQ_NTT_BLOCK_H_
#define PPQ_NTT_BLOCK_H_

/// Cooperative NTTs: T threads transform one polynomial held in shared memory (device only).
///
/// Each of the 8 layers has 128 butterflies; thread t of the group runs butterflies t, t + T,
/// ... of each layer, with a block-wide barrier between layers. The butterflies and twiddles
/// are those of ntt.h, applied in the same layer order, so the results equal ntt() and
/// invntt_tomont() bit for bit.
///
/// Every thread of the block must call these the same number of times, because of the
/// barriers: a group with no polynomial passes active = false and only takes part in them.

#include <hip/hip_runtime.h>

#include <cstdint>

#include "ppq/hd.h"
#include "ppq/ntt.h"

namespace ppq {

/// Host and device: nvcc rejects a host constexpr function even inside a static_assert in device
/// code (without --expt-relaxed-constexpr, which the build does not use)
template <unsigned T> PPQ_HD constexpr bool valid_ntt_group() {
    return T >= 1 && T <= 128 && 128 % T == 0;
}

template <unsigned T> __device__ inline void ntt_block(std::int32_t *s, unsigned t, bool active) {
    static_assert(valid_ntt_group<T>(), "T must divide 128");
    for (unsigned len = 128; len >= 1; len >>= 1) {
        if (active) {
            for (unsigned b = t; b < 128; b += T) {
                const unsigned group = b / len;
                const unsigned j = 2 * len * group + b % len;
                ct_butterfly(s[j], s[j + len], zeta(forward_zeta_index(len, group)));
            }
        }
        __syncthreads();
    }
}

template <unsigned T>
__device__ inline void invntt_tomont_block(std::int32_t *s, unsigned t, bool active) {
    static_assert(valid_ntt_group<T>(), "T must divide 128");
    for (unsigned len = 1; len < N; len <<= 1) {
        if (active) {
            for (unsigned b = t; b < 128; b += T) {
                const unsigned group = b / len;
                const unsigned j = 2 * len * group + b % len;
                gs_butterfly(s[j], s[j + len], -zeta(inverse_zeta_index(len, group)));
            }
        }
        __syncthreads();
    }
    if (active) {
        for (unsigned c = t; c < N; c += T) {
            s[c] = mont_mul(INTT_SCALE, s[c]);
        }
    }
    __syncthreads();
}

} // namespace ppq

#endif
