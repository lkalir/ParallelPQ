/// Decompose, UseHint and w1Encode (ppq/rounding.h) against the naive transcription:
/// exhaustively over every r in [0, q) and both hint values, for both gamma2

#include <array>
#include <vector>

#include "check.h"
#include "naive_mldsa.h"
#include "ppq/drbg.h"
#include "ppq/rounding.h"
#include "vector_util.h"

namespace {

using naive::Poly;

template <int P> std::size_t exhaustive_mismatches() {
    const ppq::ParamSet &p = *ppq::find_param_set(P);
    std::size_t wrong = 0;
    for (std::int32_t r = 0; r < ppq::Q; ++r) {
        std::int32_t r0 = 0;
        std::int32_t r0_naive = 0;
        std::int32_t r1_naive = 0;
        const std::int32_t r1 = ppq::decompose<P>(r, r0);
        naive::decompose(p, r, r1_naive, r0_naive);
        wrong += r1 != r1_naive || r0 != r0_naive;
        wrong += ppq::use_hint<P>(0, r) != naive::use_hint(p, 0, r);
        wrong += ppq::use_hint<P>(1, r) != naive::use_hint(p, 1, r);
    }
    return wrong;
}

TEST_CASE(decompose_and_use_hint_for_every_coefficient) {
    // ML-DSA-65 and -87 share gamma2 = (q - 1) / 32; ML-DSA-44 has (q - 1) / 88
    CHECK(exhaustive_mismatches<44>() == 0);
    CHECK(exhaustive_mismatches<65>() == 0);
}

TEST_CASE(w1_pack_matches_naive_and_the_stage_vectors) {
    for (const ppq::ParamSet &p : ppq::PARAM_SETS) {
        const std::int32_t m = (ppq::Q - 1) / (2 * p.gamma2);
        ppq::Drbg rng(0x7731, p.name);
        for (int trial = 0; trial < 50; ++trial) {
            std::vector<Poly> w1(static_cast<std::size_t>(p.k));
            for (Poly &poly : w1) {
                for (auto &c : poly) {
                    c = static_cast<std::int32_t>(rng.uniform(static_cast<std::uint64_t>(m)));
                }
            }
            std::vector<std::uint8_t> packed(p.w1_bytes);
            ppq::with_params(p.id, [&](auto mp) {
                using M = decltype(mp);
                for (int i = 0; i < M::k; ++i) {
                    ppq::w1_pack_poly<M::id>(w1[static_cast<std::size_t>(i)].data(),
                                             packed.data() + i * M::w1_poly_bytes);
                }
                return 0;
            });
            CHECK(packed == naive::w1_encode(p, w1));
        }
    }

    for (const ppq::KvRecord &r :
         ppq::load_kv_records(TEST_VECTORS_DIR "/stages/mldsa_stages.txt")) {
        if (r.str("kind") != "sig" || r.integer("hint_ok") != 1) {
            continue;
        }
        const ppq::ParamSet &p = vector_util::params_of(r);
        const auto k = static_cast<std::size_t>(p.k);
        const auto w = vector_util::polys(r, "w_approx", k);
        const auto h = vector_util::polys(r, "h", k);
        const auto w1 = vector_util::polys(r, "w1", k);
        std::vector<std::uint8_t> packed(p.w1_bytes);
        ppq::with_params(p.id, [&](auto mp) {
            using M = decltype(mp);
            for (std::size_t i = 0; i < k; ++i) {
                Poly got{};
                for (unsigned c = 0; c < ppq::N; ++c) {
                    got[c] = ppq::use_hint<M::id>(static_cast<std::uint8_t>(h[i][c]), w[i][c]);
                }
                CHECK(got == w1[i]);
                ppq::w1_pack_poly<M::id>(got.data(), packed.data() + i * M::w1_poly_bytes);
            }
            return 0;
        });
        CHECK(packed == r.bytes("w1_packed"));
    }
}

} // namespace

int main() { return check::run_all(); }
