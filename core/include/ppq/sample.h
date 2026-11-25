#ifndef PPQ_SAMPLE_H_
#define PPQ_SAMPLE_H_

/// ExpandA's rejection sampling and SampleInBall (FIPS 204 Algorithms 29, 30 and 32), for host
/// and device

#include <cstddef>
#include <cstdint>

#include "ppq/hd.h"
#include "ppq/keccak.h"
#include "ppq/mldsa_traits.h"
#include "ppq/params.h"

namespace ppq {

/// SHAKE128 output one rate-sized block at a time: the byte source RejNTTPoly reads
struct Shake128Blocks {
    Shake128 xof;

    PPQ_HD void next_block(std::uint8_t *out) { xof.squeeze(out, SHAKE128_RATE); }
};

/// RejNTTPoly (Algorithm 30) with CoeffFromThreeBytes (Algorithm 14): candidates are 3 bytes,
/// top bit dropped, kept when below q. 168 is a multiple of 3, so candidates never straddle
/// blocks. There is no bound on the number of blocks: it keeps reading until it has 256
/// coefficients. `Source` provides next_block(uint8_t[168]); tests substitute their own.
template <class Source> PPQ_HD inline void rej_ntt_poly(Source &src, std::int32_t *a) {
    static_assert(SHAKE128_RATE % 3 == 0, "candidates must not straddle blocks");
    std::uint8_t block[SHAKE128_RATE];
    unsigned j = 0;
    while (j < N) {
        src.next_block(block);
        for (unsigned b = 0; b < SHAKE128_RATE && j < N; b += 3) {
            const std::uint32_t t = static_cast<std::uint32_t>(block[b]) |
                                    (static_cast<std::uint32_t>(block[b + 1]) << 8) |
                                    (static_cast<std::uint32_t>(block[b + 2] & 0x7F) << 16);
            if (t < static_cast<std::uint32_t>(Q)) {
                a[j++] = static_cast<std::int32_t>(t);
            }
        }
    }
}

/// A_hat[row][col] of ExpandA (Algorithm 32): RejNTTPoly(rho || col || row)
PPQ_HD inline void expand_a_poly(const std::uint8_t *rho, unsigned row, unsigned col,
                                 std::int32_t *a) {
    std::uint8_t seed[SEED_BYTES + 2];
    for (std::size_t i = 0; i < SEED_BYTES; ++i) {
        seed[i] = rho[i];
    }
    seed[SEED_BYTES] = static_cast<std::uint8_t>(col);
    seed[SEED_BYTES + 1] = static_cast<std::uint8_t>(row);

    Shake128Blocks src;
    src.xof.absorb(seed, sizeof(seed));
    rej_ntt_poly(src, a);
}

/// SampleInBall (Algorithm 29): tau coefficients of +-1 from H(c-tilde). The first 8 output
/// bytes give the signs; later bytes are positions, rejected while greater than i.
template <int P> PPQ_HD inline void sample_in_ball(const std::uint8_t *ctilde, std::int32_t *c) {
    using M = Mldsa<P>;
    Shake256 h;
    h.absorb(ctilde, M::ctilde_bytes);

    std::uint8_t buf[SHAKE256_RATE];
    h.squeeze(buf, SHAKE256_RATE);
    std::uint64_t signs = load_le64(buf);
    unsigned pos = 8;

    for (unsigned i = 0; i < N; ++i) {
        c[i] = 0;
    }
    for (unsigned i = N - static_cast<unsigned>(M::tau); i < N; ++i) {
        unsigned j = 0;
        do {
            if (pos == SHAKE256_RATE) {
                h.squeeze(buf, SHAKE256_RATE);
                pos = 0;
            }
            j = buf[pos++];
        } while (j > i);
        c[i] = c[j];
        c[j] = 1 - 2 * static_cast<std::int32_t>(signs & 1);
        signs >>= 1;
    }
}

} // namespace ppq

#endif
