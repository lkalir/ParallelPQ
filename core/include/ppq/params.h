#ifndef PPQ_PARAMS_H_
#define PPQ_PARAMS_H_

/// ML-DSA (FIPS 204) constants and parameter sets.
///
/// Everything is constexpr so kernels can be templated on a parameter set and keep k, l,
/// gamma1 and omega as compile-time constants. Section and table numbers refer to FIPS 204.

#include <cstddef>
#include <cstdint>

#include "ppq/hd.h"

namespace ppq {

/// The modulus q = 2^23 - 2^13 + 1
constexpr std::int32_t Q = 8380417;
/// Coefficients per polynomial
constexpr std::size_t N = 256;
/// Bits dropped from t by Power2Round
constexpr int D = 13;

constexpr std::size_t SEED_BYTES = 32; ///< rho, and the keygen seed xi
constexpr std::size_t TR_BYTES = 64;   ///< tr = H(pk, 64)
constexpr std::size_t MU_BYTES = 64;   ///< mu = H(tr || M', 64)
constexpr std::size_t RND_BYTES = 32;  ///< signing randomness
constexpr std::size_t MAX_CONTEXT_BYTES = 255;

/// t1 coefficients have bitlen(q - 1) - d = 10 bits
constexpr int T1_BITS = 10;
constexpr std::size_t T1_PACKED_BYTES = N * T1_BITS / 8;

/// Number of bits needed to write v (bitlen in FIPS 204)
PPQ_HD constexpr int bit_length(std::uint32_t v) {
    int n = 0;
    while (v != 0) {
        ++n;
        v >>= 1;
    }
    return n;
}

struct ParamSet {
    /// 44, 65 or 87
    int id = 0;
    const char *name = "";

    // Table 1
    int k = 0;
    int l = 0;
    int eta = 0;
    int tau = 0;
    /// Collision strength of c-tilde in bits
    int lambda = 0;
    /// gamma1 = 2^gamma1_bits
    int gamma1_bits = 0;
    std::int32_t gamma2 = 0;
    int omega = 0;

    // Derived
    std::int32_t gamma1 = 0;
    /// tau * eta
    std::int32_t beta = 0;
    /// Bits per packed z coefficient: 1 + bitlen(gamma1 - 1)
    int z_bits = 0;
    /// Bits per packed w1 coefficient: bitlen((q - 1) / (2 gamma2) - 1)
    int w1_bits = 0;
    std::size_t ctilde_bytes = 0;
    std::size_t z_poly_bytes = 0;
    /// omega index bytes followed by k cumulative counts
    std::size_t hint_bytes = 0;
    std::size_t w1_bytes = 0;

    // Table 2
    std::size_t pk_bytes = 0;
    std::size_t sk_bytes = 0;
    std::size_t sig_bytes = 0;
};

constexpr ParamSet make_param_set(int id, const char *name, int k, int l, int eta, int tau,
                                  int lambda, int gamma1_bits, std::int32_t gamma2, int omega) {
    ParamSet p;
    p.id = id;
    p.name = name;
    p.k = k;
    p.l = l;
    p.eta = eta;
    p.tau = tau;
    p.lambda = lambda;
    p.gamma1_bits = gamma1_bits;
    p.gamma2 = gamma2;
    p.omega = omega;

    const auto ku = static_cast<std::size_t>(k);
    const auto lu = static_cast<std::size_t>(l);

    p.gamma1 = std::int32_t{1} << gamma1_bits;
    p.beta = tau * eta;
    p.z_bits = 1 + bit_length(static_cast<std::uint32_t>(p.gamma1 - 1));
    p.w1_bits = bit_length(static_cast<std::uint32_t>((Q - 1) / (2 * gamma2) - 1));
    p.ctilde_bytes = static_cast<std::size_t>(lambda) / 4;
    p.z_poly_bytes = N * static_cast<std::size_t>(p.z_bits) / 8;
    p.hint_bytes = static_cast<std::size_t>(omega) + ku;
    p.w1_bytes = ku * N * static_cast<std::size_t>(p.w1_bits) / 8;

    const auto eta_bits = static_cast<std::size_t>(bit_length(static_cast<std::uint32_t>(2 * eta)));
    p.pk_bytes = SEED_BYTES + ku * T1_PACKED_BYTES;
    p.sk_bytes = 2 * SEED_BYTES + TR_BYTES +
                 N / 8 * ((ku + lu) * eta_bits + ku * static_cast<std::size_t>(D));
    p.sig_bytes = p.ctilde_bytes + lu * p.z_poly_bytes + p.hint_bytes;
    return p;
}

constexpr ParamSet ML_DSA_44 =
    make_param_set(44, "ML-DSA-44", 4, 4, 2, 39, 128, 17, (Q - 1) / 88, 80);
constexpr ParamSet ML_DSA_65 =
    make_param_set(65, "ML-DSA-65", 6, 5, 4, 49, 192, 19, (Q - 1) / 32, 55);
constexpr ParamSet ML_DSA_87 =
    make_param_set(87, "ML-DSA-87", 8, 7, 2, 60, 256, 19, (Q - 1) / 32, 75);
constexpr ParamSet PARAM_SETS[] = {ML_DSA_44, ML_DSA_65, ML_DSA_87};

constexpr const ParamSet *find_param_set(int id) {
    for (const ParamSet &p : PARAM_SETS) {
        if (p.id == id) {
            return &p;
        }
    }
    return nullptr;
}

/// Largest sizes over all parameter sets, for fixed-size scratch buffers
constexpr int MAX_K = 8;
constexpr int MAX_L = 7;

// FIPS 204 Table 1
static_assert(ML_DSA_44.beta == 78 && ML_DSA_65.beta == 196 && ML_DSA_87.beta == 120);
static_assert(ML_DSA_44.gamma2 == 95232 && ML_DSA_65.gamma2 == 261888);
// FIPS 204 Table 2
static_assert(ML_DSA_44.pk_bytes == 1312 && ML_DSA_65.pk_bytes == 1952 &&
              ML_DSA_87.pk_bytes == 2592);
static_assert(ML_DSA_44.sk_bytes == 2560 && ML_DSA_65.sk_bytes == 4032 &&
              ML_DSA_87.sk_bytes == 4896);
static_assert(ML_DSA_44.sig_bytes == 2420 && ML_DSA_65.sig_bytes == 3309 &&
              ML_DSA_87.sig_bytes == 4627);
// w1Encode: 6-bit coefficients for ML-DSA-44, 4-bit otherwise
static_assert(ML_DSA_44.w1_bits == 6 && ML_DSA_65.w1_bits == 4 && ML_DSA_87.w1_bits == 4);
static_assert(ML_DSA_44.z_bits == 18 && ML_DSA_65.z_bits == 20 && ML_DSA_87.z_bits == 20);
static_assert(ML_DSA_87.k == MAX_K && ML_DSA_87.l == MAX_L);

} // namespace ppq

#endif
