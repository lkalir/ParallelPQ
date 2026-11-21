#ifndef TESTS_NTT_TEST_DATA_H_
#define TESTS_NTT_TEST_DATA_H_

/// NTT test inputs with known transforms: every polynomial in the stage vectors that the
/// verifier transforms (the NTT vectors' a, each z, c and t1 * 2^d), plus random polynomials

#include <vector>

#include "naive_mldsa.h"
#include "ppq/drbg.h"
#include "ppq/reduce.h"
#include "vector_util.h"

namespace ntt_test {

using naive::Poly;

struct Case {
    Poly input;
    /// Canonical NTT(input), from the vectors or the naive transform
    Poly expected;
};

inline std::vector<Case> forward_cases(std::size_t random_count = 16) {
    const auto records = ppq::load_kv_records(TEST_VECTORS_DIR "/stages/mldsa_stages.txt");
    std::vector<Case> out;
    for (const ppq::KvRecord &r : records) {
        const std::string &kind = r.str("kind");
        if (kind == "ntt") {
            out.push_back({vector_util::poly(r, "a"), vector_util::poly(r, "a_hat")});
        } else if (kind == "sig") {
            const auto l = static_cast<std::size_t>(vector_util::params_of(r).l);
            const auto z = vector_util::polys(r, "z", l);
            const auto z_hat = vector_util::polys(r, "z_hat", l);
            for (std::size_t j = 0; j < l; ++j) {
                out.push_back({z[j], z_hat[j]});
            }
            out.push_back({vector_util::poly(r, "c"), vector_util::poly(r, "c_hat")});
        } else if (kind == "key") {
            const auto k = static_cast<std::size_t>(vector_util::params_of(r).k);
            const auto t1 = vector_util::polys(r, "t1", k);
            const auto t1_hat = vector_util::polys(r, "t1_hat", k);
            for (std::size_t i = 0; i < k; ++i) {
                Poly shifted = t1[i];
                for (auto &c : shifted) {
                    c <<= ppq::D;
                }
                out.push_back({shifted, t1_hat[i]});
            }
        }
    }
    for (std::size_t i = 0; i < random_count; ++i) {
        ppq::Drbg rng(0x6e747467, "forward", i);
        Poly a{};
        for (auto &c : a) {
            c = static_cast<std::int32_t>(rng.uniform(2 * ppq::Q - 1)) - (ppq::Q - 1);
        }
        out.push_back({a, naive::ntt(a)});
    }
    return out;
}

/// Inputs in (-q, q) with canonical NTT^-1(input) as the expectation
inline std::vector<Case> inverse_cases(std::size_t count = 24) {
    std::vector<Case> out;
    for (std::size_t i = 0; i < count; ++i) {
        ppq::Drbg rng(0x6e747467, "inverse", i);
        Poly a{};
        for (auto &c : a) {
            c = static_cast<std::int32_t>(rng.uniform(2 * ppq::Q - 1)) - (ppq::Q - 1);
        }
        out.push_back({a, naive::intt(a)});
    }
    return out;
}

/// What a correct invntt_tomont output reduces to: remove the 2^32 factor, then canonicalise
inline std::int32_t untomont(std::int32_t c) { return ppq::caddq(ppq::montgomery_reduce(c)); }

} // namespace ntt_test

#endif
