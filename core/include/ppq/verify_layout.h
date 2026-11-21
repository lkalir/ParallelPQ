#ifndef PPQ_VERIFY_LAYOUT_H_
#define PPQ_VERIFY_LAYOUT_H_

/// Device-side data of the staged GPU verifier: plain views of device pointers that are passed
/// to kernels by value, and the per-key and per-signature strides that lay them out.
///
/// A batch is one chunk of a Workload: signatures, their key indices (into the chunk's key
/// table) and ragged messages and contexts with offsets rebased to the chunk.

#include <cstddef>
#include <cstdint>

#include "ppq/params.h"
#include "ppq/workload.h"

namespace ppq {

struct SigInputs {
    const std::uint8_t *sigs = nullptr;
    const std::uint32_t *key_index = nullptr;
    const std::uint64_t *msg_offsets = nullptr;
    const std::uint8_t *msg_data = nullptr;
    const std::uint64_t *ctx_offsets = nullptr;
    const std::uint8_t *ctx_data = nullptr;
    std::size_t count = 0;
    Interface iface = Interface::External;
};

struct KeyInputs {
    const std::uint8_t *pks = nullptr;
    std::size_t count = 0;
};

/// What each key contributes to every signature under it
struct KeyCacheView {
    /// H(pk, 64)
    std::uint8_t *tr = nullptr;
    /// ExpandA(rho), k * l polynomials, NTT domain, coefficients in [0, q)
    std::int32_t *a_hat = nullptr;
    /// NTT(t1 * 2^d), k polynomials, |a| < 9q
    std::int32_t *t1_hat = nullptr;
    std::size_t count = 0;
};

/// Per-signature intermediate values, each `stride` elements apart
struct SigWork {
    /// STATUS_* bits; 0 is a valid signature
    std::uint32_t *status = nullptr;
    /// z after decoding, then NTT(z) in place: l polynomials
    std::int32_t *z = nullptr;
    /// Hint bits, k * 256 bytes of 0 or 1
    std::uint8_t *h = nullptr;
    /// SampleInBall(c-tilde), then NTT(c) in place
    std::int32_t *c = nullptr;
    std::uint8_t *mu = nullptr;
    /// A z - c t1 2^d: NTT domain, then w_approx after the inverse NTT; k polynomials
    std::int32_t *w = nullptr;
    /// w1Encode(UseHint(h, w_approx))
    std::uint8_t *w1 = nullptr;
};

enum StatusBits : std::uint32_t {
    STATUS_HINT_BAD = 1u << 0,
    STATUS_ZNORM_BAD = 1u << 1,
    STATUS_COMMITMENT_BAD = 1u << 2,
};

/// FIPS 204 precedence: a malformed hint, then the norm bound, then the commitment
inline Verdict verdict_of(std::uint32_t status) {
    if (status & STATUS_HINT_BAD) {
        return Verdict::BadHint;
    }
    if (status & STATUS_ZNORM_BAD) {
        return Verdict::ZNorm;
    }
    return (status & STATUS_COMMITMENT_BAD) ? Verdict::BadCommitment : Verdict::Pass;
}

struct KeyStrides {
    std::size_t tr = TR_BYTES;
    std::size_t a_hat = 0;
    std::size_t t1_hat = 0;
};

struct WorkStrides {
    std::size_t z = 0;
    std::size_t h = 0;
    std::size_t c = N;
    std::size_t mu = MU_BYTES;
    std::size_t w = 0;
    std::size_t w1 = 0;
};

inline KeyStrides key_strides(const ParamSet &p) {
    KeyStrides s;
    s.a_hat = static_cast<std::size_t>(p.k * p.l) * N;
    s.t1_hat = static_cast<std::size_t>(p.k) * N;
    return s;
}

inline WorkStrides work_strides(const ParamSet &p) {
    WorkStrides s;
    s.z = static_cast<std::size_t>(p.l) * N;
    s.h = static_cast<std::size_t>(p.k) * N;
    s.w = static_cast<std::size_t>(p.k) * N;
    s.w1 = p.w1_bytes;
    return s;
}

} // namespace ppq

#endif
