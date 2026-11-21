#ifndef PPQ_REDUCE_H_
#define PPQ_REDUCE_H_

/// Arithmetic mod q = 8380417 on signed 32-bit coefficients, for host and device.
///
/// Montgomery form uses R = 2^32: montgomery_reduce(a) = a * 2^-32 mod q, so multiplying by a
/// value stored as x * 2^32 (the zeta table) multiplies by x. Bounds are stated per function
/// and are what callers must respect to avoid overflow.

#include <cstdint>

#include "ppq/hd.h"
#include "ppq/params.h"

namespace ppq {

/// q^-1 mod 2^32
constexpr std::int32_t QINV = 58728449;
/// 2^32 mod q, centred
constexpr std::int32_t MONT = -4186625;
/// 2^64 / 256 mod q: scales the inverse NTT by 1/256 and into Montgomery form
constexpr std::int32_t INTT_SCALE = 41978;

static_assert(static_cast<std::uint32_t>(QINV) * static_cast<std::uint32_t>(Q) == 1u,
              "QINV must invert q mod 2^32");

/// For |a| <= 2^31 * q, returns r = a * 2^-32 mod q with -q < r < q
PPQ_HD inline std::int32_t montgomery_reduce(std::int64_t a) {
    const auto t =
        static_cast<std::int32_t>(static_cast<std::uint32_t>(a) * static_cast<std::uint32_t>(QINV));
    return static_cast<std::int32_t>((a - static_cast<std::int64_t>(t) * Q) >> 32);
}

/// a * b * 2^-32 mod q, in (-q, q), for |a * b| <= 2^31 * q
PPQ_HD inline std::int32_t mont_mul(std::int32_t a, std::int32_t b) {
    return montgomery_reduce(static_cast<std::int64_t>(a) * b);
}

/// For a <= 2^31 - 2^22 - 1, returns r = a mod q with -6283008 <= r <= 6283008
PPQ_HD inline std::int32_t reduce32(std::int32_t a) {
    const std::int32_t t = (a + (1 << 22)) >> 23;
    return a - t * Q;
}

/// Adds q to a negative a: maps (-q, q) to [0, q)
PPQ_HD inline std::int32_t caddq(std::int32_t a) { return a + ((a >> 31) & Q); }

/// The canonical representative in [0, q), for a <= 2^31 - 2^22 - 1
PPQ_HD inline std::int32_t freeze(std::int32_t a) { return caddq(reduce32(a)); }

} // namespace ppq

#endif
