#ifndef PPQ_DECODE_H_
#define PPQ_DECODE_H_

/// Public key and signature decoding (FIPS 204 Algorithms 21, 23 and 27), for host and device.
///
/// Coefficients are packed little-endian, so byte-aligned groups decode independently:
/// t1 (10 bits) 4 per 5 bytes, z at 18 bits 4 per 9 bytes, z at 20 bits 2 per 5 bytes.

#include <cstddef>
#include <cstdint>

#include "ppq/hd.h"
#include "ppq/mldsa_traits.h"
#include "ppq/params.h"

namespace ppq {

/// One t1 polynomial from its 320 packed bytes (SimpleBitUnpack, 10 bits)
PPQ_HD inline void t1_decode_poly(const std::uint8_t *in, std::int32_t *t1) {
    for (unsigned g = 0; g < N / 4; ++g) {
        const std::uint8_t *b = in + 5 * g;
        t1[4 * g + 0] = static_cast<std::int32_t>((b[0] | (b[1] << 8)) & 0x3FF);
        t1[4 * g + 1] = static_cast<std::int32_t>(((b[1] >> 2) | (b[2] << 6)) & 0x3FF);
        t1[4 * g + 2] = static_cast<std::int32_t>(((b[2] >> 4) | (b[3] << 4)) & 0x3FF);
        t1[4 * g + 3] = static_cast<std::int32_t>(((b[3] >> 6) | (b[4] << 2)) & 0x3FF);
    }
}

/// Little-endian bytes [first, first + n) as an integer, n <= 8
PPQ_HD inline std::uint64_t load_bytes_le(const std::uint8_t *p, unsigned n) {
    std::uint64_t v = 0;
    for (unsigned i = 0; i < n; ++i) {
        v |= static_cast<std::uint64_t>(p[i]) << (8 * i);
    }
    return v;
}

/// One z polynomial (BitUnpack(., gamma1 - 1, gamma1)): z = gamma1 - packed value
template <int P> PPQ_HD inline void z_decode_poly(const std::uint8_t *in, std::int32_t *z) {
    using M = Mldsa<P>;
    constexpr unsigned bits = static_cast<unsigned>(M::z_bits);
    static_assert(bits == 18 || bits == 20, "z is packed in 18 or 20 bits");
    constexpr unsigned per_group = bits == 18 ? 4 : 2;
    constexpr unsigned group_bytes = per_group * bits / 8;
    constexpr std::uint64_t mask = (std::uint64_t{1} << bits) - 1;

    for (unsigned g = 0; g < N / per_group; ++g) {
        // A 9-byte group does not fit 64 bits: take the first 8, and the last byte separately
        const std::uint8_t *b = in + group_bytes * g;
        const std::uint64_t lo = load_bytes_le(b, group_bytes < 8 ? group_bytes : 8);
        for (unsigned i = 0; i < per_group; ++i) {
            const unsigned shift = i * bits;
            std::uint64_t v = lo >> shift;
            if (shift + bits > 64) {
                v |= static_cast<std::uint64_t>(b[8]) << (64 - shift);
            }
            z[per_group * g + i] = M::gamma1 - static_cast<std::int32_t>(v & mask);
        }
    }
}

/// ||z||_inf < gamma1 - beta for one polynomial
template <int P> PPQ_HD inline bool z_poly_norm_ok(const std::int32_t *z) {
    constexpr std::int32_t bound = Mldsa<P>::gamma1 - Mldsa<P>::beta;
    bool ok = true;
    for (unsigned c = 0; c < N; ++c) {
        const std::int32_t a = z[c] < 0 ? -z[c] : z[c];
        ok = ok && a < bound;
    }
    return ok;
}

/// HintBitUnpack (Algorithm 21) of the omega + k hint bytes into k * 256 bits (one byte each).
/// Returns false for a malformed encoding (the algorithm's bottom); h is then unspecified.
template <int P> PPQ_HD inline bool hint_decode(const std::uint8_t *y, std::uint8_t *h) {
    using M = Mldsa<P>;
    for (unsigned c = 0; c < static_cast<unsigned>(M::k) * N; ++c) {
        h[c] = 0;
    }

    unsigned index = 0;
    for (unsigned i = 0; i < static_cast<unsigned>(M::k); ++i) {
        const unsigned end = y[M::omega + i];
        if (end < index || end > static_cast<unsigned>(M::omega)) {
            return false;
        }
        const unsigned first = index;
        for (; index < end; ++index) {
            // Indices within a polynomial must be strictly increasing
            if (index > first && y[index - 1] >= y[index]) {
                return false;
            }
            h[i * N + y[index]] = 1;
        }
    }
    // Unused index slots must be zero
    for (unsigned i = index; i < static_cast<unsigned>(M::omega); ++i) {
        if (y[i] != 0) {
            return false;
        }
    }
    return true;
}

} // namespace ppq

#endif
