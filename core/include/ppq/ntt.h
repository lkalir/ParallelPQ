#ifndef PPQ_NTT_H_
#define PPQ_NTT_H_

/// The ML-DSA number-theoretic transform over Z_q[X]/(X^256 + 1), for host and device.
///
/// ntt() is FIPS 204 Algorithm 41 (Cooley-Tukey, output in bit-reversed order);
/// invntt_tomont() is Algorithm 42 (Gentleman-Sande) with an extra factor of 2^32, so that
/// it cancels the 2^-32 a Montgomery pointwise product leaves behind. Neither reduces
/// between layers:
///
///   ntt:            |a| < q in   ->  |a| < 9q out
///   invntt_tomont:  |a| < q in   ->  |a| < q out, equal to NTT^-1(a) * 2^32 mod q
///
/// These are the per-thread transforms (one thread owns a whole polynomial). The butterflies
/// are separate so the cooperative kernels can share them.

#include <cstdint>

#include "ppq/hd.h"
#include "ppq/ntt_tables.h"
#include "ppq/params.h"
#include "ppq/reduce.h"

namespace ppq {

/// Montgomery-form twiddle factor k (see ntt_tables.h)
PPQ_HD inline std::int32_t zeta(unsigned k) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return detail::ZETAS_DEVICE[k];
#else
    return detail::ZETAS[k];
#endif
}

/// Cooley-Tukey butterfly: (a, b) -> (a + z b, a - z b)
PPQ_HD inline void ct_butterfly(std::int32_t &a, std::int32_t &b, std::int32_t z) {
    const std::int32_t t = mont_mul(z, b);
    b = a - t;
    a = a + t;
}

/// Gentleman-Sande butterfly: (a, b) -> (a + b, z (a - b))
PPQ_HD inline void gs_butterfly(std::int32_t &a, std::int32_t &b, std::int32_t z) {
    const std::int32_t t = a;
    a = t + b;
    b = mont_mul(z, t - b);
}

/// Layer with half-width `len` has 128 / len groups; group g of the forward transform uses
/// zeta(128 / len + g) and of the inverse uses -zeta(256 / len - 1 - g)
PPQ_HD inline unsigned forward_zeta_index(unsigned len, unsigned group) {
    return 128 / len + group;
}
PPQ_HD inline unsigned inverse_zeta_index(unsigned len, unsigned group) {
    return 256 / len - 1 - group;
}

/// Forward NTT in place
PPQ_HD inline void ntt(std::int32_t *a) {
    for (unsigned len = 128; len >= 1; len >>= 1) {
        for (unsigned group = 0; group < 128 / len; ++group) {
            const std::int32_t z = zeta(forward_zeta_index(len, group));
            const unsigned start = 2 * len * group;
            for (unsigned j = start; j < start + len; ++j) {
                ct_butterfly(a[j], a[j + len], z);
            }
        }
    }
}

/// Inverse NTT in place, times 2^32
PPQ_HD inline void invntt_tomont(std::int32_t *a) {
    for (unsigned len = 1; len < N; len <<= 1) {
        for (unsigned group = 0; group < 128 / len; ++group) {
            const std::int32_t z = -zeta(inverse_zeta_index(len, group));
            const unsigned start = 2 * len * group;
            for (unsigned j = start; j < start + len; ++j) {
                gs_butterfly(a[j], a[j + len], z);
            }
        }
    }
    for (unsigned j = 0; j < N; ++j) {
        a[j] = mont_mul(INTT_SCALE, a[j]);
    }
}

} // namespace ppq

#endif
