#ifndef PPQ_DRBG_H_
#define PPQ_DRBG_H_

/// Deterministic byte streams for reproducible workloads.
///
/// Each stream is SHAKE256("ppq-drbg-v1" || le64(seed) || len(label) || label || le64(index)),
/// so the bytes for, say, signature 1234 depend only on the seed and that index, never on
/// thread count or generation order. Not for producing real keys.

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "ppq/keccak.h"

namespace ppq {

class Drbg {
  public:
    Drbg(std::uint64_t seed, std::string_view label, std::uint64_t index = 0) {
        static constexpr char DOMAIN[] = "ppq-drbg-v1";
        std::uint8_t word[8];

        _xof.absorb(reinterpret_cast<const std::uint8_t *>(DOMAIN), sizeof(DOMAIN) - 1);
        store_le64(word, seed);
        _xof.absorb(word, sizeof(word));

        const auto label_len = static_cast<std::uint8_t>(label.size() < 255 ? label.size() : 255);
        _xof.absorb(&label_len, 1);
        _xof.absorb(reinterpret_cast<const std::uint8_t *>(label.data()), label_len);

        store_le64(word, index);
        _xof.absorb(word, sizeof(word));
    }

    void fill(std::uint8_t *out, std::size_t n) { _xof.squeeze(out, n); }

    std::uint64_t next_u64() {
        std::uint8_t word[8];
        fill(word, sizeof(word));
        return load_le64(word);
    }

    /// Uniform in [0, bound), without modulo bias; bound 0 returns 0
    std::uint64_t uniform(std::uint64_t bound) {
        if (bound == 0) {
            return 0;
        }
        // Largest multiple of bound that fits, minus one
        const std::uint64_t limit = UINT64_MAX - UINT64_MAX % bound;
        std::uint64_t v = 0;
        do {
            v = next_u64();
        } while (v >= limit);
        return v % bound;
    }

    /// Uniform in [lo, hi]
    std::uint64_t range(std::uint64_t lo, std::uint64_t hi) {
        return hi <= lo ? lo : lo + uniform(hi - lo + 1);
    }

  private:
    Shake256 _xof;
};

} // namespace ppq

#endif
