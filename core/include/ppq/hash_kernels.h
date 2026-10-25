#ifndef PPQ_HASH_KERNELS_H_
#define PPQ_HASH_KERNELS_H_

#include <hip/hip_runtime.h>

#include "ppq/launch.h"
#include "ppq/mldsa.h"

namespace ppq {

/// Most threads per block both hashing kernels can launch on the current device
int hash_kernels_max_threads();

/// Kernel 1: one thread per signature computes tr and mu (SHAKE256)
void launch_hash_mu(const BatchLayout &layout, const BatchView &v, const Launch &launch,
                    hipStream_t stream);

/// Kernel 2: one thread per (signature, i, j) squeezes an ExpandA stream (SHAKE128)
void launch_expand_a(const BatchLayout &layout, const BatchView &v, const Launch &launch,
                     hipStream_t stream);

} // namespace ppq

#endif
