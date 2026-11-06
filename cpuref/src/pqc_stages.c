/* ML-DSA verification broken into stages with the pq-crystals reference internals.
 *
 * Compiled once per mode with DILITHIUM_MODE = 2, 3 or 5 against the pq-crystals ref
 * headers, so K, L, poly_ntt() etc. resolve to that mode. Follows crypto_sign_verify_internal
 * step for step, but keeps every intermediate value and converts NTT-domain values out of
 * pq-crystals' unreduced representation into canonical form. */

#include <string.h>

#include "fips202.h"
#include "packing.h"
#include "params.h"
#include "poly.h"
#include "polyvec.h"
#include "reduce.h"

#include "pqc_stages.h"

#if DILITHIUM_MODE == 2
#define PPQ_FN(name) ppq_pqc2_##name
#elif DILITHIUM_MODE == 3
#define PPQ_FN(name) ppq_pqc3_##name
#elif DILITHIUM_MODE == 5
#define PPQ_FN(name) ppq_pqc5_##name
#else
#error "DILITHIUM_MODE must be 2, 3 or 5"
#endif

_Static_assert(K <= PPQ_PQC_MAX_K && L <= PPQ_PQC_MAX_L, "stage arrays too small");
_Static_assert(K *POLYW1_PACKEDBYTES <= 1024, "w1 buffer too small");

static int32_t canonical(int32_t a) {
    a %= Q;
    return a < 0 ? a + Q : a;
}

static void store_canonical(int32_t *out, const poly *p) {
    for (unsigned i = 0; i < N; ++i) {
        out[i] = canonical(p->coeffs[i]);
    }
}

static void store_raw(int32_t *out, const poly *p) { memcpy(out, p->coeffs, sizeof(p->coeffs)); }

void PPQ_FN(key_stages)(ppq_pqc_key_stages *out, const uint8_t *pk) {
    polyveck t1;
    polyvecl mat[K];

    memset(out, 0, sizeof(*out));
    unpack_pk(out->rho, &t1, pk);
    shake256(out->tr, TRBYTES, pk, CRYPTO_PUBLICKEYBYTES);

    for (unsigned i = 0; i < K; ++i) {
        store_raw(out->t1[i], &t1.vec[i]);
    }

    polyvec_matrix_expand(mat, out->rho);
    for (unsigned i = 0; i < K; ++i) {
        for (unsigned j = 0; j < L; ++j) {
            store_canonical(out->a_hat[i][j], &mat[i].vec[j]);
        }
    }

    polyveck_shiftl(&t1);
    polyveck_ntt(&t1);
    for (unsigned i = 0; i < K; ++i) {
        store_canonical(out->t1_hat[i], &t1.vec[i]);
    }
}

void PPQ_FN(mu)(uint8_t *mu, const uint8_t *pk, const uint8_t *pre, size_t prelen,
                const uint8_t *msg, size_t msglen) {
    uint8_t tr[TRBYTES];
    keccak_state state;

    shake256(tr, TRBYTES, pk, CRYPTO_PUBLICKEYBYTES);
    shake256_init(&state);
    shake256_absorb(&state, tr, TRBYTES);
    shake256_absorb(&state, pre, prelen);
    shake256_absorb(&state, msg, msglen);
    shake256_finalize(&state);
    shake256_squeeze(mu, CRHBYTES, &state);
}

void PPQ_FN(sig_stages)(ppq_pqc_sig_stages *out, const uint8_t *pk, const uint8_t *mu,
                        const uint8_t *sig) {
    uint8_t rho[SEEDBYTES];
    poly cp;
    polyvecl mat[K], z;
    polyveck t1, w, h;
    keccak_state state;

    memset(out, 0, sizeof(*out));
    unpack_pk(rho, &t1, pk);

    /* unpack_sig decodes c-tilde and z before the hint, so both are valid even when it fails */
    out->hint_ok = unpack_sig(out->ctilde, &z, &h, sig) == 0;
    out->znorm_ok = polyvecl_chknorm(&z, GAMMA1 - BETA) == 0;

    for (unsigned j = 0; j < L; ++j) {
        store_raw(out->z[j], &z.vec[j]);
    }
    if (out->hint_ok) {
        for (unsigned i = 0; i < K; ++i) {
            store_raw(out->h[i], &h.vec[i]);
        }
    }

    poly_challenge(&cp, out->ctilde);
    store_raw(out->c, &cp);

    polyvecl_ntt(&z);
    for (unsigned j = 0; j < L; ++j) {
        store_canonical(out->z_hat[j], &z.vec[j]);
    }
    poly_ntt(&cp);
    store_canonical(out->c_hat, &cp);

    if (!out->hint_ok) {
        return;
    }

    /* w_approx = NTT^-1(A_hat o z_hat - c_hat o NTT(t1 * 2^d)); the Montgomery factors of the
     * pointwise products and invntt_tomont cancel */
    polyvec_matrix_expand(mat, rho);
    polyvec_matrix_pointwise_montgomery(&w, mat, &z);
    polyveck_shiftl(&t1);
    polyveck_ntt(&t1);
    polyveck_pointwise_poly_montgomery(&t1, &cp, &t1);
    polyveck_sub(&w, &w, &t1);
    polyveck_reduce(&w);
    polyveck_invntt_tomont(&w);
    polyveck_caddq(&w);
    for (unsigned i = 0; i < K; ++i) {
        store_canonical(out->w_approx[i], &w.vec[i]);
    }

    polyveck_use_hint(&w, &w, &h);
    for (unsigned i = 0; i < K; ++i) {
        store_raw(out->w1[i], &w.vec[i]);
    }
    polyveck_pack_w1(out->w1_packed, &w);

    shake256_init(&state);
    shake256_absorb(&state, mu, CRHBYTES);
    shake256_absorb(&state, out->w1_packed, K * POLYW1_PACKEDBYTES);
    shake256_finalize(&state);
    shake256_squeeze(out->ctilde_prime, CTILDEBYTES, &state);

    out->commitment_ok = memcmp(out->ctilde, out->ctilde_prime, CTILDEBYTES) == 0;
}

void PPQ_FN(ntt)(int32_t *a) {
    poly p;
    memcpy(p.coeffs, a, sizeof(p.coeffs));
    poly_ntt(&p);
    store_canonical(a, &p);
}

/* invntt_tomont multiplies by the Montgomery factor 2^32; one more Montgomery reduction
 * divides it back out, leaving the plain FIPS 204 NTT^-1 */
void PPQ_FN(intt)(int32_t *a) {
    poly p;
    memcpy(p.coeffs, a, sizeof(p.coeffs));
    poly_invntt_tomont(&p);
    for (unsigned i = 0; i < N; ++i) {
        p.coeffs[i] = montgomery_reduce((int64_t)p.coeffs[i]);
    }
    store_canonical(a, &p);
}
