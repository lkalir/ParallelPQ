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

} // namespace ppq

#endif
