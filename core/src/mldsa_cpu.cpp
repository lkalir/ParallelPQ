#include "ppq/mldsa.h"

#include <random>

namespace ppq {

void fill_random_inputs(const BatchLayout &layout, std::uint8_t *pk, std::uint8_t *msg,
                        std::size_t count, std::uint32_t seed) {
    std::mt19937_64 rng(seed);

    const auto fill = [&](std::uint8_t *p, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            p[i] = static_cast<std::uint8_t>(rng());
        }
    };

    fill(pk, count * layout.pk_stride());

    for (std::size_t s = 0; s < count; ++s) {
        std::uint8_t *m = msg + s * layout.msg_stride();
        m[0] = 0;
        m[1] = 0;
        fill(m + MESSAGE_PREFIX_BYTES, layout.msg_bytes);
    }
}

void hash_mu_cpu(const BatchLayout &layout, const BatchView &v) {
    for (std::size_t s = 0; s < v.count; ++s) {
        hash_mu_one(v.pk + s * layout.pk_stride(), layout.pk_stride(),
                    v.msg + s * layout.msg_stride(), layout.msg_stride(),
                    v.digest + s * layout.digest_stride());
    }
}

void expand_a_cpu(const BatchLayout &layout, const BatchView &v) {
    const auto &p = layout.params;

    for (std::size_t s = 0; s < v.count; ++s) {
        std::uint8_t *out = v.matrix + s * layout.matrix_stride();

        for (int i = 0; i < p.k; ++i) {
            for (int j = 0; j < p.l; ++j) {
                expand_a_one(v.pk + s * layout.pk_stride(), i, j, out);
                out += EXPAND_A_BYTES;
            }
        }
    }
}

} // namespace ppq
