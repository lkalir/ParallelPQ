#ifndef PPQ_VERIFY_KERNELS_H_
#define PPQ_VERIFY_KERNELS_H_

/// The stages of the staged GPU verifier. Each launcher takes the runtime parameter set,
/// dispatches to kernels templated on it, and uses `threads` threads per block with enough
/// blocks to cover the work (grid-stride loops handle the rest).
///
/// Stages run in order on one stream; each reads what earlier stages wrote to SigWork.

#include <hip/hip_runtime.h>

#include "ppq/params.h"
#include "ppq/verify_layout.h"

namespace ppq {

/// Most threads per block every stage kernel can use on the current device
int verify_kernels_max_threads();

/// Stage 1, decode: clears the status words, decodes z into work.z (one thread per signature
/// and z polynomial) and the hint into work.h (one thread per signature), and sets
/// STATUS_ZNORM_BAD and STATUS_HINT_BAD.
void launch_decode(const ParamSet &p, const SigInputs &in, const SigWork &work, unsigned threads,
                   hipStream_t stream);

/// Key stage, tr: tr = H(pk, 64) for every key, one thread per key
void launch_key_tr(const ParamSet &p, const KeyInputs &keys, const KeyCacheView &cache,
                   unsigned threads, hipStream_t stream);

/// Stage 2, mu: mu for every signature through the batch's interface, from its key's tr in the
/// cache and its ragged message and context; one thread per signature
void launch_mu(const ParamSet &p, const SigInputs &in, const KeyCacheView &cache,
               const SigWork &work, unsigned threads, hipStream_t stream);

/// Key stage, ExpandA: A_hat for every key by rejection sampling, one thread per (key, row,
/// column) entry
void launch_key_expand_a(const ParamSet &p, const KeyInputs &keys, const KeyCacheView &cache,
                         unsigned threads, hipStream_t stream);

/// Stage 3, challenge: c = SampleInBall(c-tilde) and then NTT(c) in place in work.c, one thread
/// per signature
void launch_challenge(const ParamSet &p, const SigInputs &in, const SigWork &work, unsigned threads,
                      hipStream_t stream);

/// Key stage, t1: NTT(t1 * 2^d) for every key, one thread per (key, polynomial)
void launch_key_t1(const ParamSet &p, const KeyInputs &keys, const KeyCacheView &cache,
                   unsigned threads, hipStream_t stream);

/// Fills the key cache: tr, A_hat and NTT(t1 * 2^d) for every key. Runs once per key table,
/// however many signatures share each key.
void launch_prepare_keys(const ParamSet &p, const KeyInputs &keys, const KeyCacheView &cache,
                         unsigned threads, hipStream_t stream);

/// Stage 4, NTT(z): transforms work.z in place with the cooperative NTT, `ntt_group` threads
/// per polynomial (see launch_ntt_coop; threads must be a multiple of it)
void launch_z_ntt(const ParamSet &p, const SigInputs &in, const SigWork &work, unsigned ntt_group,
                  unsigned threads, hipStream_t stream);

/// Stage 5, the matrix-vector product: w = sum_j A_hat[i][j] o z_hat[j] - c_hat o t1_hat[i] in
/// the NTT domain, from each signature's key in the cache; one thread per (signature, row,
/// coefficient). The Montgomery factors of the products are cancelled by the inverse NTT.
void launch_matvec(const ParamSet &p, const SigInputs &in, const KeyCacheView &cache,
                   const SigWork &work, unsigned threads, hipStream_t stream);

/// Stage 6, NTT^-1(w): work.w in place, giving w_approx with |w| < q
void launch_w_intt(const ParamSet &p, const SigInputs &in, const SigWork &work, unsigned ntt_group,
                   unsigned threads, hipStream_t stream);

/// Stage 7, UseHint and w1Encode: w1 = UseHint(h, w_approx) packed into work.w1, one thread per
/// (signature, row)
void launch_use_hint(const ParamSet &p, const SigInputs &in, const SigWork &work, unsigned threads,
                     hipStream_t stream);

/// Stage 8, the commitment: c-tilde' = H(mu || w1Encode(w1), lambda / 4) compared with the
/// signature's c-tilde, setting STATUS_COMMITMENT_BAD; one thread per signature
void launch_final_hash(const ParamSet &p, const SigInputs &in, const SigWork &work,
                       unsigned threads, hipStream_t stream);

} // namespace ppq

#endif
