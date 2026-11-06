#ifndef PPQ_CPUREF_STAGES_H_
#define PPQ_CPUREF_STAGES_H_

/// The intermediate values of ML-DSA verification (FIPS 204 Algorithm 8), computed with the
/// pq-crystals reference internals. These are the oracle for ParallelPQ's GPU stages and the
/// source of tests/vectors/stages.
///
/// Polynomials are canonical (coefficients in [0, q)) except z (signed, in (-gamma1, gamma1]),
/// c (in {-1, 0, 1}) and h (0 or 1). Vectors of polynomials are row-major: a_hat[i * l + j].

#include <array>
#include <cstdint>
#include <vector>

#include "ppq/cpuref.h"
#include "ppq/workload.h"

namespace ppq::cpuref {

using Poly = std::array<std::int32_t, N>;

struct KeyStages {
    Bytes rho;
    std::vector<Poly> t1;     ///< k
    Bytes tr;                 ///< H(pk, 64)
    std::vector<Poly> a_hat;  ///< k * l, ExpandA(rho)
    std::vector<Poly> t1_hat; ///< k, NTT(t1 * 2^d)
};

struct SigStages {
    Verdict verdict = Verdict::BadCommitment;
    bool hint_ok = false;
    Bytes ctilde;
    std::vector<Poly> z;     ///< l
    std::vector<Poly> h;     ///< k; empty when the hint is malformed
    Poly c{};                ///< SampleInBall(c-tilde)
    std::vector<Poly> z_hat; ///< l
    Poly c_hat{};
    // Empty when the hint is malformed:
    std::vector<Poly> w_approx; ///< k, NTT^-1(A_hat o z_hat - c_hat o t1_hat)
    std::vector<Poly> w1;       ///< k, UseHint(h, w_approx)
    Bytes w1_packed;            ///< w1Encode(w1)
    Bytes ctilde_prime;         ///< H(mu || w1Encode(w1), lambda / 4)
};

KeyStages key_stages(const ParamSet &params, ByteView pk);

/// From mu, so it covers every interface; mu_external / mu_internal compute mu
SigStages sig_stages(const ParamSet &params, ByteView pk, ByteView mu, ByteView sig);

/// mu = H(tr || 0 || len(ctx) || ctx || msg, 64), for ML-DSA.Verify
Bytes mu_external(const ParamSet &params, ByteView pk, ByteView msg, ByteView ctx);

/// mu = H(tr || M', 64), for ML-DSA.Verify_internal
Bytes mu_internal(const ParamSet &params, ByteView pk, ByteView mprime);

/// FIPS 204 NTT and NTT^-1 (Algorithms 41 and 42) on canonical coefficients
Poly ntt(const Poly &a);
Poly intt(const Poly &a_hat);

} // namespace ppq::cpuref

#endif
