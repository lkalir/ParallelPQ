#ifndef PPQ_CPUREF_PQC_STAGES_H_
#define PPQ_CPUREF_PQC_STAGES_H_

/* Intermediate values of ML-DSA verification, computed with the pq-crystals reference
 * internals (pqc_stages.c, compiled once per mode). Sized for the largest parameter set;
 * only the first k / l rows are meaningful. Polynomials are canonical: coefficients in
 * [0, q), except z (signed, in (-gamma1, gamma1]), c (in {-1, 0, 1}) and h (0 or 1). */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PPQ_PQC_N 256
#define PPQ_PQC_MAX_K 8
#define PPQ_PQC_MAX_L 7

typedef struct {
    uint8_t rho[32];
    int32_t t1[PPQ_PQC_MAX_K][PPQ_PQC_N];
    uint8_t tr[64];
    /* ExpandA(rho), NTT domain */
    int32_t a_hat[PPQ_PQC_MAX_K][PPQ_PQC_MAX_L][PPQ_PQC_N];
    /* NTT(t1 * 2^d) */
    int32_t t1_hat[PPQ_PQC_MAX_K][PPQ_PQC_N];
} ppq_pqc_key_stages;

typedef struct {
    /* HintBitUnpack succeeded; when 0 the fields from w_approx on are not computed */
    int hint_ok;
    /* ||z||_inf < gamma1 - beta */
    int znorm_ok;
    /* c-tilde == c-tilde' */
    int commitment_ok;
    uint8_t ctilde[64];
    int32_t z[PPQ_PQC_MAX_L][PPQ_PQC_N];
    int32_t h[PPQ_PQC_MAX_K][PPQ_PQC_N];
    /* SampleInBall(c-tilde) */
    int32_t c[PPQ_PQC_N];
    int32_t z_hat[PPQ_PQC_MAX_L][PPQ_PQC_N];
    int32_t c_hat[PPQ_PQC_N];
    /* NTT^-1(A_hat o z_hat - c_hat o t1_hat) */
    int32_t w_approx[PPQ_PQC_MAX_K][PPQ_PQC_N];
    /* UseHint(h, w_approx) */
    int32_t w1[PPQ_PQC_MAX_K][PPQ_PQC_N];
    uint8_t w1_packed[1024];
    uint8_t ctilde_prime[64];
} ppq_pqc_sig_stages;

#define PPQ_PQC_DECLARE_STAGES(m)                                                                  \
    void ppq_pqc##m##_key_stages(ppq_pqc_key_stages *out, const uint8_t *pk);                      \
    void ppq_pqc##m##_sig_stages(ppq_pqc_sig_stages *out, const uint8_t *pk, const uint8_t *mu,    \
                                 const uint8_t *sig);                                              \
    void ppq_pqc##m##_mu(uint8_t *mu, const uint8_t *pk, const uint8_t *pre, size_t prelen,        \
                         const uint8_t *msg, size_t msglen);                                       \
    void ppq_pqc##m##_ntt(int32_t *a);                                                             \
    void ppq_pqc##m##_intt(int32_t *a);

PPQ_PQC_DECLARE_STAGES(2)
PPQ_PQC_DECLARE_STAGES(3)
PPQ_PQC_DECLARE_STAGES(5)

#ifdef __cplusplus
}
#endif

#endif
