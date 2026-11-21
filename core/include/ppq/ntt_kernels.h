#ifndef PPQ_NTT_KERNELS_H_
#define PPQ_NTT_KERNELS_H_

/// Batched NTTs over polynomials in device memory (count * 256 int32 coefficients, in place).
/// These standalone kernels test and benchmark the transforms; the verifier calls the same
/// device functions from its own kernels.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "ppq/launch.h"

namespace ppq {

enum class NttDirection {
    /// ntt(): |a| < q in, |a| < 9q out
    Forward,
    /// invntt_tomont(): |a| < q in, |a| < q out, times 2^32
    Inverse,
};

/// Most threads per block launch_ntt_simple can use on the current device
int ntt_simple_max_threads();

/// One thread per polynomial, transforming it in place in global memory (grid-stride over
/// polynomials). The baseline every other NTT kernel must match bit for bit.
void launch_ntt_simple(std::int32_t *polys, std::size_t count, NttDirection dir,
                       const Launch &launch, hipStream_t stream);

/// Threads per polynomial launch_ntt_coop supports (each must divide 128)
constexpr unsigned NTT_GROUP_SIZES[] = {16, 32, 64, 128};

bool valid_ntt_group_size(unsigned group);

/// Most threads per block launch_ntt_coop can use with this group size
int ntt_coop_max_threads(unsigned group);

/// `group` threads cooperate on each polynomial in shared memory (ntt_block.h). launch.threads
/// must be a multiple of `group`; each block then holds launch.threads / group polynomials
/// (1 KiB of shared memory each) and strides over the batch. Results equal launch_ntt_simple
/// bit for bit. Throws std::invalid_argument for an unsupported group or block size.
void launch_ntt_coop(std::int32_t *polys, std::size_t count, NttDirection dir, unsigned group,
                     const Launch &launch, hipStream_t stream);

} // namespace ppq

#endif
