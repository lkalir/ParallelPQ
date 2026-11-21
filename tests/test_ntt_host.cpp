/// Modular arithmetic and the per-thread NTT (ppq/reduce.h, ppq/ntt.h) on the host, against
/// the stage vectors and the naive FIPS 204 transforms

#include <algorithm>
#include <cstdlib>

#include "check.h"
#include "naive_mldsa.h"
#include "ppq/drbg.h"
#include "ppq/ntt.h"
#include "vector_util.h"

namespace {

using naive::Poly;
using vector_util::params_of;
using vector_util::poly;
using vector_util::polys;

constexpr std::uint64_t TEST_SEED = 0x6e7474; // "ntt"
constexpr std::int64_t Q = ppq::Q;

const std::vector<ppq::KvRecord> &stages() {
    static const auto r = ppq::load_kv_records(TEST_VECTORS_DIR "/stages/mldsa_stages.txt");
    return r;
}

Poly frozen(Poly a) {
    for (auto &c : a) {
        c = ppq::freeze(c);
    }
    return a;
}

Poly forward(Poly a) {
    ppq::ntt(a.data());
    return a;
}

/// invntt_tomont leaves a factor 2^32; one Montgomery reduction removes it
Poly inverse(Poly a) {
    ppq::invntt_tomont(a.data());
    for (auto &c : a) {
        c = ppq::caddq(ppq::montgomery_reduce(c));
    }
    return a;
}

Poly random_poly(std::uint64_t index, std::int32_t lo, std::int32_t hi) {
    ppq::Drbg rng(TEST_SEED, "poly", index);
    Poly a{};
    for (auto &c : a) {
        c = lo + static_cast<std::int32_t>(rng.uniform(static_cast<std::uint64_t>(hi - lo + 1)));
    }
    return a;
}

std::int64_t mod_q(std::int64_t a) { return ((a % Q) + Q) % Q; }

TEST_CASE(constants_are_consistent) {
    const std::int64_t r = (std::int64_t{1} << 32) % Q;
    CHECK(mod_q(ppq::MONT) == r);
    CHECK(mod_q(ppq::INTT_SCALE) == naive::pow_mod(2, 56)); // 2^64 / 2^8
    CHECK(static_cast<std::uint32_t>(ppq::QINV) * static_cast<std::uint32_t>(ppq::Q) == 1u);
}

TEST_CASE(zeta_table_is_montgomery_powers_of_1753) {
    for (unsigned k = 0; k < 256; ++k) {
        const std::int64_t want =
            mod_q(naive::pow_mod(naive::ZETA, naive::bitrev8(k)) * ((std::int64_t{1} << 32) % Q));
        CHECK(mod_q(ppq::zeta(k)) == want);
        CHECK(std::abs(ppq::zeta(k)) <= Q / 2);
    }
}

TEST_CASE(reductions_match_modular_arithmetic) {
    ppq::Drbg rng(TEST_SEED, "reduce");
    const std::int64_t inv_r = naive::pow_mod((std::int64_t{1} << 32) % Q, Q - 2);
    for (int i = 0; i < 20000; ++i) {
        const auto a = static_cast<std::int32_t>(rng.uniform(2 * 9 * Q)) - 9 * ppq::Q;
        const auto b = static_cast<std::int32_t>(rng.uniform(2 * Q)) - ppq::Q;

        const std::int32_t m = ppq::mont_mul(a, b);
        CHECK(m > -ppq::Q && m < ppq::Q);
        CHECK(mod_q(m) == mod_q(mod_q(a) * mod_q(b) % Q * inv_r));

        const auto big = static_cast<std::int32_t>(rng.uniform(std::uint64_t{1} << 31) -
                                                   (std::uint64_t{1} << 30));
        const std::int32_t r = ppq::reduce32(big);
        CHECK(r >= -6283008 && r <= 6283008);
        CHECK(mod_q(r) == mod_q(big));
        CHECK(ppq::freeze(big) == mod_q(big));
    }
    CHECK(ppq::caddq(-1) == ppq::Q - 1);
    CHECK(ppq::caddq(0) == 0);
    CHECK(ppq::freeze(ppq::Q) == 0);
}

TEST_CASE(ntt_matches_the_vectors) {
    std::size_t checked = 0;
    for (const ppq::KvRecord &r : stages()) {
        const std::string &kind = r.str("kind");
        if (kind == "ntt") {
            CHECK(frozen(forward(poly(r, "a"))) == poly(r, "a_hat"));
            CHECK(inverse(poly(r, "a_hat")) == poly(r, "a"));
            ++checked;
        } else if (kind == "sig") {
            const ppq::ParamSet &p = params_of(r);
            const auto z = polys(r, "z", static_cast<std::size_t>(p.l));
            const auto z_hat = polys(r, "z_hat", static_cast<std::size_t>(p.l));
            for (std::size_t j = 0; j < z.size(); ++j) {
                CHECK(frozen(forward(z[j])) == z_hat[j]);
            }
            CHECK(frozen(forward(poly(r, "c"))) == poly(r, "c_hat"));
            ++checked;
        } else if (kind == "key") {
            const ppq::ParamSet &p = params_of(r);
            const auto t1 = polys(r, "t1", static_cast<std::size_t>(p.k));
            const auto t1_hat = polys(r, "t1_hat", static_cast<std::size_t>(p.k));
            for (std::size_t i = 0; i < t1.size(); ++i) {
                Poly shifted = t1[i];
                for (auto &c : shifted) {
                    c <<= ppq::D;
                }
                CHECK(frozen(forward(shifted)) == t1_hat[i]);
            }
            ++checked;
        }
    }
    CHECK(checked == stages().size());
}

TEST_CASE(transforms_match_the_naive_definition_and_bounds) {
    for (std::uint64_t i = 0; i < 6; ++i) {
        // Inputs anywhere in the allowed range (-q, q)
        const Poly a = random_poly(i, -ppq::Q + 1, ppq::Q - 1);
        const Poly a_hat = forward(a);
        CHECK(frozen(a_hat) == naive::ntt(a));
        CHECK(std::all_of(a_hat.begin(), a_hat.end(),
                          [](std::int32_t c) { return std::abs(c) < 9 * ppq::Q; }));

        const Poly b = random_poly(100 + i, -ppq::Q + 1, ppq::Q - 1);
        Poly b_inv = b;
        ppq::invntt_tomont(b_inv.data());
        CHECK(std::all_of(b_inv.begin(), b_inv.end(),
                          [](std::int32_t c) { return std::abs(c) < ppq::Q; }));
        CHECK(inverse(b) == naive::intt(b));

        // The round trip through both transforms is the identity (mod q, times 2^32)
        CHECK(inverse(frozen(a_hat)) == frozen(a));
    }
}

TEST_CASE(pointwise_products_cancel_the_montgomery_factors) {
    // NTT^-1(NTT(a) o NTT(b)) via mont_mul then invntt_tomont gives the negacyclic product
    const Poly a = random_poly(200, -ppq::Q + 1, ppq::Q - 1);
    const Poly b = random_poly(201, -1, 1);
    const Poly ah = forward(a);
    const Poly bh = forward(b);
    Poly prod{};
    for (unsigned c = 0; c < ppq::N; ++c) {
        prod[c] = ppq::mont_mul(ah[c], bh[c]);
    }
    ppq::invntt_tomont(prod.data());

    Poly want{};
    for (unsigned i = 0; i < ppq::N; ++i) {
        for (unsigned j = 0; j < ppq::N; ++j) {
            const std::int64_t t = static_cast<std::int64_t>(a[i]) * b[j];
            const unsigned k = (i + j) % ppq::N;
            want[k] = static_cast<std::int32_t>(mod_q(want[k] + (i + j < ppq::N ? t : -t)));
        }
    }
    CHECK(frozen(prod) == want);
}

} // namespace

int main() { return check::run_all(); }
