#ifndef PPQ_KECCAK_H_
#define PPQ_KECCAK_H_

#include <cstddef>
#include <cstdint>

#include "ppq/hd.h"

namespace ppq {

constexpr unsigned KECCAK_LANES = 25;
constexpr unsigned KECCAK_ROUNDS = 24;
constexpr unsigned SHAKE128_RATE = 168;
constexpr unsigned SHAKE256_RATE = 136;
constexpr std::uint8_t SHAKE_DOMAIN = 0x1F;
constexpr std::uint8_t PAD_LAST_BIT = 0x80;

PPQ_HD inline std::uint64_t rotl64(std::uint64_t x, unsigned n) {
    return (x << n) | (x >> (64 - n));
}

/// FIPS 202 Keccak-p[1600, 24]
PPQ_HD inline void keccak_f1600(std::uint64_t a[KECCAK_LANES]) {
    const std::uint64_t round_constants[KECCAK_ROUNDS] = {
        0x0000000000000001ull, 0x0000000000008082ull, 0x800000000000808aull, 0x8000000080008000ull,
        0x000000000000808bull, 0x0000000080000001ull, 0x8000000080008081ull, 0x8000000000008009ull,
        0x000000000000008aull, 0x0000000000000088ull, 0x0000000080008009ull, 0x000000008000000aull,
        0x000000008000808bull, 0x800000000000008bull, 0x8000000000008089ull, 0x8000000000008003ull,
        0x8000000000008002ull, 0x8000000000000080ull, 0x000000000000800aull, 0x800000008000000aull,
        0x8000000080008081ull, 0x8000000000008080ull, 0x0000000080000001ull, 0x8000000080008008ull,
    };
    // rho offsets and pi destinations, in the order the rho-pi walk visits lanes
    const unsigned rho[24] = {1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
                              27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44};
    const unsigned pi[24] = {10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
                             15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1};

    PPQ_UNROLL
    for (unsigned round = 0; round < KECCAK_ROUNDS; ++round) {
        std::uint64_t c[5];

        PPQ_UNROLL
        for (unsigned x = 0; x < 5; ++x) {
            c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
        }

        PPQ_UNROLL
        for (unsigned x = 0; x < 5; ++x) {
            const std::uint64_t d = c[(x + 4) % 5] ^ rotl64(c[(x + 1) % 5], 1);
            PPQ_UNROLL
            for (unsigned y = 0; y < 25; y += 5) {
                a[y + x] ^= d;
            }
        }

        std::uint64_t carry = a[1];
        PPQ_UNROLL
        for (unsigned i = 0; i < 24; ++i) {
            const std::uint64_t next = a[pi[i]];
            a[pi[i]] = rotl64(carry, rho[i]);
            carry = next;
        }

        PPQ_UNROLL
        for (unsigned y = 0; y < 25; y += 5) {
            PPQ_UNROLL
            for (unsigned x = 0; x < 5; ++x) {
                c[x] = a[y + x];
            }
            PPQ_UNROLL
            for (unsigned x = 0; x < 5; ++x) {
                a[y + x] = c[x] ^ (~c[(x + 1) % 5] & c[(x + 2) % 5]);
            }
        }

        a[0] ^= round_constants[round];
    }
}

PPQ_HD inline std::uint64_t load_le64(const std::uint8_t *p) {
    std::uint64_t v = 0;
    PPQ_UNROLL
    for (unsigned i = 0; i < 8; ++i) {
        v |= static_cast<std::uint64_t>(p[i]) << (8 * i);
    }
    return v;
}

PPQ_HD inline void store_le64(std::uint8_t *p, std::uint64_t v) {
    PPQ_UNROLL
    for (unsigned i = 0; i < 8; ++i) {
        p[i] = static_cast<std::uint8_t>(v >> (8 * i));
    }
}

/// SHAKE sponge. The state is only ever indexed with compile-time lane numbers
/// so a GPU compiler can keep it in registers; partial blocks go through `_buf`.
template <unsigned Rate> class Sponge {
    static_assert(Rate % 8 == 0 && Rate < 8 * KECCAK_LANES, "rate must be whole lanes");

  public:
    static constexpr unsigned RATE = Rate;

    PPQ_HD Sponge() {
        PPQ_UNROLL
        for (unsigned i = 0; i < KECCAK_LANES; ++i) {
            _state[i] = 0;
        }
    }

    /// Returns false (and does nothing) once squeezing has started
    PPQ_HD bool absorb(const std::uint8_t *in, std::size_t len) {
        if (_squeezing) {
            return false;
        }

        while (len > 0) {
            if (_pos == 0 && len >= Rate) {
                absorb_block(in);
                in += Rate;
                len -= Rate;
                continue;
            }

            const std::size_t room = Rate - _pos;
            const std::size_t n = len < room ? len : room;
            for (std::size_t i = 0; i < n; ++i) {
                _buf[_pos + i] = in[i];
            }
            _pos += static_cast<unsigned>(n);
            in += n;
            len -= n;

            if (_pos == Rate) {
                absorb_block(_buf);
                _pos = 0;
            }
        }

        return true;
    }

    PPQ_HD void finalize() {
        if (_squeezing) {
            return;
        }

        for (unsigned i = _pos; i < Rate; ++i) {
            _buf[i] = 0;
        }
        _buf[_pos] = SHAKE_DOMAIN;
        _buf[Rate - 1] |= PAD_LAST_BIT;
        absorb_block(_buf);
        store_block();
        _pos = 0;
        _squeezing = true;
    }

    PPQ_HD void squeeze(std::uint8_t *out, std::size_t len) {
        finalize();

        while (len > 0) {
            if (_pos == Rate) {
                keccak_f1600(_state);
                store_block();
                _pos = 0;
            }

            const std::size_t avail = Rate - _pos;
            const std::size_t n = len < avail ? len : avail;
            for (std::size_t i = 0; i < n; ++i) {
                out[i] = _buf[_pos + i];
            }
            _pos += static_cast<unsigned>(n);
            out += n;
            len -= n;
        }
    }

  private:
    PPQ_HD void absorb_block(const std::uint8_t *block) {
        PPQ_UNROLL
        for (unsigned i = 0; i < Rate / 8; ++i) {
            _state[i] ^= load_le64(block + 8 * i);
        }
        keccak_f1600(_state);
    }

    PPQ_HD void store_block() {
        PPQ_UNROLL
        for (unsigned i = 0; i < Rate / 8; ++i) {
            store_le64(_buf + 8 * i, _state[i]);
        }
    }

    std::uint64_t _state[KECCAK_LANES];
    std::uint8_t _buf[Rate];
    unsigned _pos = 0;
    bool _squeezing = false;
};

using Shake128 = Sponge<SHAKE128_RATE>;
using Shake256 = Sponge<SHAKE256_RATE>;

template <class Xof>
PPQ_HD inline void shake(const std::uint8_t *in, std::size_t in_len, std::uint8_t *out,
                         std::size_t out_len) {
    Xof xof;
    xof.absorb(in, in_len);
    xof.squeeze(out, out_len);
}

} // namespace ppq

#endif
