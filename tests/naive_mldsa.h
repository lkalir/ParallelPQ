#ifndef TESTS_NAIVE_MLDSA_H_
#define TESTS_NAIVE_MLDSA_H_

/// A deliberately naive ML-DSA verifier for tests, transcribed from FIPS 204 with no
/// optimisation: O(n^2) NTTs evaluated straight from the definition, bit-at-a-time packing,
/// and the algorithm numbers in the comments. It shares nothing with pq-crystals or with
/// ParallelPQ's kernels except the CAVP-tested Keccak sponge, so it is an independent check
/// on both.

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "ppq/keccak.h"
#include "ppq/params.h"
#include "ppq/workload.h"

namespace naive {

using Poly = std::array<std::int32_t, ppq::N>;
using Bytes = std::vector<std::uint8_t>;

constexpr std::int64_t Q = ppq::Q;
/// A primitive 512th root of unity mod q
constexpr std::int64_t ZETA = 1753;

inline std::int64_t mod_q(std::int64_t a) {
    a %= Q;
    return a < 0 ? a + Q : a;
}

inline std::int64_t pow_mod(std::int64_t base, std::int64_t e) {
    std::int64_t r = 1;
    base = mod_q(base);
    while (e > 0) {
        if (e & 1) {
            r = r * base % Q;
        }
        base = base * base % Q;
        e >>= 1;
    }
    return r;
}

inline unsigned bitrev8(unsigned i) {
    unsigned r = 0;
    for (int b = 0; b < 8; ++b) {
        r |= ((i >> b) & 1u) << (7 - b);
    }
    return r;
}

/// zeta^e for e in [0, 512)
inline const std::array<std::int64_t, 512> &zeta_powers() {
    static const std::array<std::int64_t, 512> table = [] {
        std::array<std::int64_t, 512> t{};
        t[0] = 1;
        for (std::size_t e = 1; e < t.size(); ++e) {
            t[e] = t[e - 1] * ZETA % Q;
        }
        return t;
    }();
    return table;
}

/// NTT (Algorithm 41) by definition: a_hat[i] = a(zeta^(2 bitrev8(i) + 1))
inline Poly ntt(const Poly &a) {
    const auto &zp = zeta_powers();
    Poly out{};
    for (unsigned i = 0; i < ppq::N; ++i) {
        const unsigned e = 2 * bitrev8(i) + 1;
        std::int64_t acc = 0;
        for (unsigned j = 0; j < ppq::N; ++j) {
            acc = (acc + mod_q(a[j]) * zp[(e * j) % 512]) % Q;
        }
        out[i] = static_cast<std::int32_t>(acc);
    }
    return out;
}

/// NTT^-1 (Algorithm 42) by definition: the inverse of ntt()
inline Poly intt(const Poly &a_hat) {
    const auto &zp = zeta_powers();
    const std::int64_t inv_n = pow_mod(ppq::N, Q - 2);
    Poly out{};
    for (unsigned j = 0; j < ppq::N; ++j) {
        std::int64_t acc = 0;
        for (unsigned i = 0; i < ppq::N; ++i) {
            const unsigned e = ((2 * bitrev8(i) + 1) * j) % 512;
            acc = (acc + mod_q(a_hat[i]) * zp[(512 - e) % 512]) % Q;
        }
        out[j] = static_cast<std::int32_t>(acc * inv_n % Q);
    }
    return out;
}

/// Little-endian bit field (BitsToInteger over BytesToBits)
inline std::uint32_t get_bits(const std::uint8_t *data, std::size_t bit, int nbits) {
    std::uint32_t v = 0;
    for (int b = 0; b < nbits; ++b) {
        const std::size_t at = bit + static_cast<std::size_t>(b);
        v |= static_cast<std::uint32_t>((data[at / 8] >> (at % 8)) & 1u) << b;
    }
    return v;
}

inline void put_bits(std::uint8_t *data, std::size_t bit, int nbits, std::uint32_t v) {
    for (int b = 0; b < nbits; ++b) {
        const std::size_t at = bit + static_cast<std::size_t>(b);
        data[at / 8] = static_cast<std::uint8_t>(data[at / 8] | (((v >> b) & 1u) << (at % 8)));
    }
}

/// pkDecode (Algorithm 23): t1 coefficients, 10 bits each after rho
inline std::vector<Poly> t1_decode(const ppq::ParamSet &p, const Bytes &pk) {
    std::vector<Poly> t1(static_cast<std::size_t>(p.k));
    for (std::size_t i = 0; i < t1.size(); ++i) {
        for (unsigned c = 0; c < ppq::N; ++c) {
            const std::size_t bit = (ppq::SEED_BYTES + i * ppq::T1_PACKED_BYTES) * 8 + c * 10;
            t1[i][c] = static_cast<std::int32_t>(get_bits(pk.data(), bit, 10));
        }
    }
    return t1;
}

/// The z part of sigDecode (Algorithm 27): BitUnpack(., gamma1 - 1, gamma1)
inline std::vector<Poly> z_decode(const ppq::ParamSet &p, const Bytes &sig) {
    std::vector<Poly> z(static_cast<std::size_t>(p.l));
    for (std::size_t j = 0; j < z.size(); ++j) {
        for (unsigned c = 0; c < ppq::N; ++c) {
            const std::size_t bit =
                (p.ctilde_bytes + j * p.z_poly_bytes) * 8 + c * static_cast<std::size_t>(p.z_bits);
            z[j][c] = p.gamma1 - static_cast<std::int32_t>(get_bits(sig.data(), bit, p.z_bits));
        }
    }
    return z;
}

/// HintBitUnpack (Algorithm 21); nullopt is the algorithm's bottom
inline std::optional<std::vector<Poly>> hint_decode(const ppq::ParamSet &p, const Bytes &sig) {
    const std::uint8_t *y = sig.data() + p.sig_bytes - p.hint_bytes;
    const auto omega = static_cast<std::size_t>(p.omega);
    std::vector<Poly> h(static_cast<std::size_t>(p.k), Poly{});
    std::size_t index = 0;

    for (std::size_t i = 0; i < h.size(); ++i) {
        if (y[omega + i] < index || y[omega + i] > omega) {
            return std::nullopt;
        }
        const std::size_t first = index;
        while (index < y[omega + i]) {
            if (index > first && y[index - 1] >= y[index]) {
                return std::nullopt;
            }
            h[i][y[index]] = 1;
            ++index;
        }
    }
    for (std::size_t i = index; i < omega; ++i) {
        if (y[i] != 0) {
            return std::nullopt;
        }
    }
    return h;
}

inline Bytes shake256(const Bytes &in, std::size_t out_len) {
    ppq::Shake256 s;
    s.absorb(in.data(), in.size());
    Bytes out(out_len);
    s.squeeze(out.data(), out.size());
    return out;
}

inline Bytes concat(std::initializer_list<Bytes> parts) {
    Bytes out;
    for (const Bytes &b : parts) {
        out.insert(out.end(), b.begin(), b.end());
    }
    return out;
}

inline Bytes tr_of(const Bytes &pk) { return shake256(pk, ppq::TR_BYTES); }

/// mu for ML-DSA.Verify (Algorithms 3 and 8): H(tr || 0 || |ctx| || ctx || M, 64)
inline Bytes mu_external(const Bytes &pk, const Bytes &msg, const Bytes &ctx) {
    const Bytes prefix = {0, static_cast<std::uint8_t>(ctx.size())};
    return shake256(concat({tr_of(pk), prefix, ctx, msg}), ppq::MU_BYTES);
}

/// SampleInBall (Algorithm 29)
inline Poly sample_in_ball(const ppq::ParamSet &p, const Bytes &ctilde) {
    ppq::Shake256 s;
    s.absorb(ctilde.data(), ctilde.size());
    std::uint8_t signs_bytes[8];
    s.squeeze(signs_bytes, 8);
    std::uint64_t signs = 0;
    for (int b = 0; b < 8; ++b) {
        signs |= static_cast<std::uint64_t>(signs_bytes[b]) << (8 * b);
    }

    Poly c{};
    for (unsigned i = ppq::N - static_cast<unsigned>(p.tau); i < ppq::N; ++i) {
        std::uint8_t j = 0;
        do {
            s.squeeze(&j, 1);
        } while (j > i);
        c[i] = c[j];
        c[j] = ((signs >> (i + static_cast<unsigned>(p.tau) - ppq::N)) & 1u) ? -1 : 1;
    }
    return c;
}

/// RejNTTPoly (Algorithm 30) with CoeffFromThreeBytes (Algorithm 14)
inline Poly rej_ntt_poly(const Bytes &seed) {
    ppq::Shake128 g;
    g.absorb(seed.data(), seed.size());
    Poly a{};
    unsigned j = 0;
    while (j < ppq::N) {
        std::uint8_t b[3];
        g.squeeze(b, 3);
        const std::uint32_t z = (static_cast<std::uint32_t>(b[2] & 0x7F) << 16) |
                                (static_cast<std::uint32_t>(b[1]) << 8) | b[0];
        if (z < static_cast<std::uint32_t>(Q)) {
            a[j++] = static_cast<std::int32_t>(z);
        }
    }
    return a;
}

/// ExpandA (Algorithm 32), row-major k x l
inline std::vector<Poly> expand_a(const ppq::ParamSet &p, const Bytes &rho) {
    std::vector<Poly> a;
    for (int r = 0; r < p.k; ++r) {
        for (int s = 0; s < p.l; ++s) {
            a.push_back(rej_ntt_poly(
                concat({rho, Bytes{static_cast<std::uint8_t>(s), static_cast<std::uint8_t>(r)}})));
        }
    }
    return a;
}

/// Decompose (Algorithm 36): r = r1 * 2 gamma2 + r0
inline void decompose(const ppq::ParamSet &p, std::int32_t r, std::int32_t &r1, std::int32_t &r0) {
    const std::int64_t alpha = 2 * static_cast<std::int64_t>(p.gamma2);
    const std::int64_t rp = mod_q(r);
    std::int64_t m = rp % alpha; // r0 = r+ mod+- alpha, in (-alpha/2, alpha/2]
    if (m > alpha / 2) {
        m -= alpha;
    }
    if (rp - m == Q - 1) {
        r1 = 0;
        r0 = static_cast<std::int32_t>(m - 1);
    } else {
        r1 = static_cast<std::int32_t>((rp - m) / alpha);
        r0 = static_cast<std::int32_t>(m);
    }
}

/// UseHint (Algorithm 40)
inline std::int32_t use_hint(const ppq::ParamSet &p, std::int32_t h, std::int32_t r) {
    const std::int32_t m = (ppq::Q - 1) / (2 * p.gamma2);
    std::int32_t r1 = 0;
    std::int32_t r0 = 0;
    decompose(p, r, r1, r0);
    if (h == 1 && r0 > 0) {
        return (r1 + 1) % m;
    }
    if (h == 1 && r0 <= 0) {
        return (r1 - 1 + m) % m;
    }
    return r1;
}

/// w1Encode (Algorithm 28)
inline Bytes w1_encode(const ppq::ParamSet &p, const std::vector<Poly> &w1) {
    Bytes out(p.w1_bytes, 0);
    std::size_t bit = 0;
    for (const Poly &poly : w1) {
        for (const std::int32_t c : poly) {
            put_bits(out.data(), bit, p.w1_bits, static_cast<std::uint32_t>(c));
            bit += static_cast<std::size_t>(p.w1_bits);
        }
    }
    return out;
}

inline std::int32_t max_abs(const std::vector<Poly> &v) {
    std::int32_t m = 0;
    for (const Poly &p : v) {
        for (const std::int32_t c : p) {
            m = std::max(m, c < 0 ? -c : c);
        }
    }
    return m;
}

/// Every stage of Verify_internal (Algorithm 8) given mu
struct Stages {
    ppq::Verdict verdict = ppq::Verdict::BadCommitment;
    Bytes rho, tr, ctilde;
    std::vector<Poly> t1, a_hat, t1_hat, z, z_hat;
    std::optional<std::vector<Poly>> h;
    Poly c{}, c_hat{};
    std::vector<Poly> w_approx, w1;
    Bytes w1_packed, ctilde_prime;
};

inline Stages verify_stages(const ppq::ParamSet &p, const Bytes &pk, const Bytes &mu,
                            const Bytes &sig) {
    Stages s;
    s.rho.assign(pk.begin(), pk.begin() + ppq::SEED_BYTES);
    s.t1 = t1_decode(p, pk);
    s.tr = tr_of(pk);
    s.a_hat = expand_a(p, s.rho);
    for (const Poly &t : s.t1) {
        Poly shifted{};
        for (unsigned c = 0; c < ppq::N; ++c) {
            shifted[c] = t[c] << ppq::D;
        }
        s.t1_hat.push_back(ntt(shifted));
    }

    s.ctilde.assign(sig.begin(), sig.begin() + static_cast<std::ptrdiff_t>(p.ctilde_bytes));
    s.z = z_decode(p, sig);
    s.h = hint_decode(p, sig);
    s.c = sample_in_ball(p, s.ctilde);
    for (const Poly &zj : s.z) {
        s.z_hat.push_back(ntt(zj));
    }
    s.c_hat = ntt(s.c);

    if (!s.h) {
        s.verdict = ppq::Verdict::BadHint;
        return s;
    }

    const auto k = static_cast<std::size_t>(p.k);
    const auto l = static_cast<std::size_t>(p.l);
    for (std::size_t i = 0; i < k; ++i) {
        Poly acc{};
        for (unsigned c = 0; c < ppq::N; ++c) {
            std::int64_t v = 0;
            for (std::size_t j = 0; j < l; ++j) {
                v += static_cast<std::int64_t>(s.a_hat[i * l + j][c]) * s.z_hat[j][c] % Q;
            }
            v -= static_cast<std::int64_t>(s.c_hat[c]) * s.t1_hat[i][c] % Q;
            acc[c] = static_cast<std::int32_t>(mod_q(v));
        }
        s.w_approx.push_back(intt(acc));

        Poly w1{};
        for (unsigned c = 0; c < ppq::N; ++c) {
            w1[c] = use_hint(p, (*s.h)[i][c], s.w_approx[i][c]);
        }
        s.w1.push_back(w1);
    }

    s.w1_packed = w1_encode(p, s.w1);
    s.ctilde_prime = shake256(concat({mu, s.w1_packed}), p.ctilde_bytes);

    if (max_abs(s.z) >= p.gamma1 - p.beta) {
        s.verdict = ppq::Verdict::ZNorm;
    } else {
        s.verdict = s.ctilde == s.ctilde_prime ? ppq::Verdict::Pass : ppq::Verdict::BadCommitment;
    }
    return s;
}

} // namespace naive

#endif
