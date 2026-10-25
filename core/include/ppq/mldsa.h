#ifndef PPQ_MLDSA_H_
#define PPQ_MLDSA_H_

#include <cstddef>
#include <cstdint>

#include "ppq/hd.h"
#include "ppq/keccak.h"

namespace ppq {

struct ParamSet {
    int id;
    int k;
    int l;
    std::size_t pk_bytes;
};

constexpr std::size_t SEED_BYTES = 32;
constexpr std::size_t T1_PACKED_BYTES = 320;
constexpr std::size_t TR_BYTES = 64;
constexpr std::size_t MU_BYTES = 64;
constexpr std::size_t DIGEST_BYTES = TR_BYTES + MU_BYTES;
/// Pure ML-DSA with an empty context: M' = 0x00 || len(ctx) = 0x00 || M
constexpr std::size_t MESSAGE_PREFIX_BYTES = 2;
/// RejNTTPoly needs at most this many SHAKE128 blocks in practice (as in the reference code)
constexpr std::size_t EXPAND_A_BLOCKS = 5;
constexpr std::size_t EXPAND_A_BYTES = EXPAND_A_BLOCKS * SHAKE128_RATE;

constexpr ParamSet make_param_set(int id, int k, int l) {
    return {id, k, l, SEED_BYTES + T1_PACKED_BYTES * static_cast<std::size_t>(k)};
}

constexpr ParamSet ML_DSA_44 = make_param_set(44, 4, 4);
constexpr ParamSet ML_DSA_65 = make_param_set(65, 6, 5);
constexpr ParamSet ML_DSA_87 = make_param_set(87, 8, 7);
constexpr ParamSet PARAM_SETS[] = {ML_DSA_44, ML_DSA_65, ML_DSA_87};

static_assert(ML_DSA_44.pk_bytes == 1312 && ML_DSA_65.pk_bytes == 1952 &&
              ML_DSA_87.pk_bytes == 2592);

constexpr const ParamSet *find_param_set(int id) {
    for (const ParamSet &p : PARAM_SETS) {
        if (p.id == id) {
            return &p;
        }
    }
    return nullptr;
}

/// Per-signature strides of the four packed arrays in a batch
struct BatchLayout {
    ParamSet params = ML_DSA_44;
    std::size_t msg_bytes = 0;

    PPQ_HD constexpr std::size_t pk_stride() const { return params.pk_bytes; }
    PPQ_HD constexpr std::size_t msg_stride() const { return MESSAGE_PREFIX_BYTES + msg_bytes; }
    PPQ_HD constexpr std::size_t digest_stride() const { return DIGEST_BYTES; }
    PPQ_HD constexpr std::size_t matrix_entries() const {
        return static_cast<std::size_t>(params.k) * static_cast<std::size_t>(params.l);
    }
    PPQ_HD constexpr std::size_t matrix_stride() const { return matrix_entries() * EXPAND_A_BYTES; }
    PPQ_HD constexpr std::size_t input_stride() const { return pk_stride() + msg_stride(); }
    PPQ_HD constexpr std::size_t output_stride() const { return digest_stride() + matrix_stride(); }
};

/// `count` signatures; digest = tr || mu, matrix = k*l ExpandA streams, row-major
struct BatchView {
    const std::uint8_t *pk = nullptr;
    const std::uint8_t *msg = nullptr;
    std::uint8_t *digest = nullptr;
    std::uint8_t *matrix = nullptr;
    std::size_t count = 0;
};

PPQ_HD constexpr BatchView slice(const BatchLayout &layout, const BatchView &v, std::size_t first,
                                 std::size_t count) {
    return {v.pk + first * layout.pk_stride(), v.msg + first * layout.msg_stride(),
            v.digest + first * layout.digest_stride(), v.matrix + first * layout.matrix_stride(),
            count};
}

/// tr = H(pk, 64), mu = H(tr || M', 64)
PPQ_HD inline void hash_mu_one(const std::uint8_t *pk, std::size_t pk_len, const std::uint8_t *msg,
                               std::size_t msg_len, std::uint8_t *digest) {
    shake<Shake256>(pk, pk_len, digest, TR_BYTES);

    Shake256 h;
    h.absorb(digest, TR_BYTES);
    h.absorb(msg, msg_len);
    h.squeeze(digest + TR_BYTES, MU_BYTES);
}

/// The SHAKE128 stream ExpandA samples A[i][j] from: G(rho || j || i)
PPQ_HD inline void expand_a_one(const std::uint8_t *rho, int i, int j, std::uint8_t *out) {
    std::uint8_t seed[SEED_BYTES + 2];

    for (std::size_t b = 0; b < SEED_BYTES; ++b) {
        seed[b] = rho[b];
    }
    seed[SEED_BYTES] = static_cast<std::uint8_t>(j);
    seed[SEED_BYTES + 1] = static_cast<std::uint8_t>(i);

    shake<Shake128>(seed, sizeof(seed), out, EXPAND_A_BYTES);
}

/// Random keys and messages; writes the M' prefix
void fill_random_inputs(const BatchLayout &layout, std::uint8_t *pk, std::uint8_t *msg,
                        std::size_t count, std::uint32_t seed);

void hash_mu_cpu(const BatchLayout &layout, const BatchView &v);
void expand_a_cpu(const BatchLayout &layout, const BatchView &v);

} // namespace ppq

#endif
