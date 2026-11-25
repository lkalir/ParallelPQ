#ifndef PPQ_ROUNDING_H_
#define PPQ_ROUNDING_H_

/// Decompose, UseHint and w1Encode (FIPS 204 Algorithms 28, 36 and 40), for host and device.
///
/// Written directly from the definitions; the divisions are by compile-time constants, which
/// compilers turn into multiplications.

#include <cstdint>

#include "ppq/hd.h"
#include "ppq/mldsa_traits.h"
#include "ppq/params.h"

namespace ppq {

/// Decompose (Algorithm 36) for r in [0, q): r = r1 * 2 gamma2 + r0 with r0 in
/// (-gamma2, gamma2], except that r - r0 = q - 1 maps to r1 = 0, r0 = r0 - 1
template <int P> PPQ_HD inline std::int32_t decompose(std::int32_t r, std::int32_t &r0) {
    constexpr std::int32_t alpha = 2 * Mldsa<P>::gamma2;
    std::int32_t m = r % alpha;
    if (m > alpha / 2) {
        m -= alpha;
    }
    if (r - m == Q - 1) {
        r0 = m - 1;
        return 0;
    }
    r0 = m;
    return (r - m) / alpha;
}

/// UseHint (Algorithm 40) for r in [0, q) and h in {0, 1}
template <int P> PPQ_HD inline std::int32_t use_hint(std::uint8_t h, std::int32_t r) {
    constexpr std::int32_t m = (Q - 1) / (2 * Mldsa<P>::gamma2);
    std::int32_t r0 = 0;
    const std::int32_t r1 = decompose<P>(r, r0);
    if (h == 0) {
        return r1;
    }
    if (r0 > 0) {
        return r1 + 1 == m ? 0 : r1 + 1;
    }
    return r1 == 0 ? m - 1 : r1 - 1;
}

/// One polynomial of w1Encode (Algorithm 28): SimpleBitPack of coefficients below
/// (q - 1) / (2 gamma2), 6 bits (4 per 3 bytes) for ML-DSA-44 and 4 bits (2 per byte) otherwise
template <int P> PPQ_HD inline void w1_pack_poly(const std::int32_t *w1, std::uint8_t *out) {
    constexpr int bits = Mldsa<P>::w1_bits;
    static_assert(bits == 4 || bits == 6, "w1 is packed in 4 or 6 bits");
    if constexpr (bits == 6) {
        for (unsigned g = 0; g < N / 4; ++g) {
            const std::uint32_t v = static_cast<std::uint32_t>(w1[4 * g]) |
                                    (static_cast<std::uint32_t>(w1[4 * g + 1]) << 6) |
                                    (static_cast<std::uint32_t>(w1[4 * g + 2]) << 12) |
                                    (static_cast<std::uint32_t>(w1[4 * g + 3]) << 18);
            out[3 * g] = static_cast<std::uint8_t>(v);
            out[3 * g + 1] = static_cast<std::uint8_t>(v >> 8);
            out[3 * g + 2] = static_cast<std::uint8_t>(v >> 16);
        }
    } else {
        for (unsigned g = 0; g < N / 2; ++g) {
            out[g] = static_cast<std::uint8_t>(w1[2 * g] | (w1[2 * g + 1] << 4));
        }
    }
}

} // namespace ppq

#endif
