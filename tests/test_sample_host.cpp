/// ExpandA and SampleInBall (ppq/sample.h) on the host: stage and ACVP vectors, the naive
/// transcription, and rejection-heavy byte sources

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

#include "check.h"
#include "naive_mldsa.h"
#include "ppq/acvp.h"
#include "ppq/drbg.h"
#include "ppq/sample.h"
#include "vector_util.h"

namespace {

using naive::Bytes;
using naive::Poly;

/// Hands out scripted blocks, then falls back to a real SHAKE128 stream
struct ScriptedBlocks {
    std::vector<std::array<std::uint8_t, ppq::SHAKE128_RATE>> script;
    std::size_t next = 0;
    ppq::Shake128Blocks fallback;
    std::size_t blocks_read = 0;

    void next_block(std::uint8_t *out) {
        ++blocks_read;
        if (next < script.size()) {
            std::memcpy(out, script[next++].data(), ppq::SHAKE128_RATE);
        } else {
            fallback.next_block(out);
        }
    }
};

Poly sample_c(const ppq::ParamSet &p, const Bytes &ctilde) {
    return ppq::with_params(p.id, [&](auto m) {
        Poly c{};
        ppq::sample_in_ball<decltype(m)::id>(ctilde.data(), c.data());
        return c;
    });
}

TEST_CASE(expand_a_matches_the_stage_vectors) {
    std::size_t polys = 0;
    for (const ppq::KvRecord &r :
         ppq::load_kv_records(TEST_VECTORS_DIR "/stages/mldsa_stages.txt")) {
        if (r.str("kind") != "key") {
            continue;
        }
        const ppq::ParamSet &p = vector_util::params_of(r);
        const Bytes rho = r.bytes("rho");
        const auto want = vector_util::polys(r, "a_hat", static_cast<std::size_t>(p.k * p.l));
        for (int i = 0; i < p.k; ++i) {
            for (int j = 0; j < p.l; ++j) {
                Poly a{};
                ppq::expand_a_poly(rho.data(), static_cast<unsigned>(i), static_cast<unsigned>(j),
                                   a.data());
                CHECK(a == want[static_cast<std::size_t>(i * p.l + j)]);
                ++polys;
            }
        }
    }
    CHECK(polys == 2 * (16 + 30 + 56));
}

TEST_CASE(sample_in_ball_matches_vectors_and_naive) {
    for (const ppq::KvRecord &r :
         ppq::load_kv_records(TEST_VECTORS_DIR "/stages/mldsa_stages.txt")) {
        if (r.str("kind") == "sig") {
            CHECK(sample_c(vector_util::params_of(r), r.bytes("ctilde")) ==
                  vector_util::poly(r, "c"));
        }
    }
    for (const ppq::AcvpWorkload &g :
         ppq::load_acvp_sigver(TEST_VECTORS_DIR "/acvp/mldsa_sigver.txt")) {
        const ppq::Workload &w = g.workload;
        for (std::size_t i = 0; i < w.sig_count(); ++i) {
            const Bytes ctilde(w.sig(i).begin(), w.sig(i).begin() + static_cast<std::ptrdiff_t>(
                                                                        w.params.ctilde_bytes));
            const Poly c = sample_c(w.params, ctilde);
            CHECK(c == naive::sample_in_ball(w.params, ctilde));
            CHECK(std::count_if(c.begin(), c.end(), [](std::int32_t v) { return v != 0; }) ==
                  w.params.tau);
        }
    }
}

TEST_CASE(expand_a_matches_naive_for_random_seeds) {
    for (const ppq::ParamSet &p : ppq::PARAM_SETS) {
        Bytes rho(ppq::SEED_BYTES);
        ppq::Drbg(0x616861, "rho", static_cast<std::uint64_t>(p.id)).fill(rho.data(), rho.size());
        const auto want = naive::expand_a(p, rho);
        for (int i = 0; i < p.k; ++i) {
            for (int j = 0; j < p.l; ++j) {
                Poly a{};
                ppq::expand_a_poly(rho.data(), static_cast<unsigned>(i), static_cast<unsigned>(j),
                                   a.data());
                CHECK(a == want[static_cast<std::size_t>(i * p.l + j)]);
            }
        }
    }
}

/// Every candidate in an all-0xFF block is 0x7FFFFF >= q, so whole blocks are rejected: the
/// sampler must keep squeezing past the 5 blocks module 6 assumed
TEST_CASE(fully_rejected_blocks_are_skipped) {
    std::array<std::uint8_t, ppq::SHAKE128_RATE> reject{};
    reject.fill(0xFF);

    ScriptedBlocks scripted;
    scripted.script.assign(7, reject);
    const std::uint8_t seed[3] = {1, 2, 3};
    scripted.fallback.xof.absorb(seed, sizeof(seed));

    ppq::Shake128Blocks plain;
    plain.xof.absorb(seed, sizeof(seed));

    Poly a{};
    Poly b{};
    ppq::rej_ntt_poly(scripted, a.data());
    ppq::rej_ntt_poly(plain, b.data());
    CHECK(a == b);
    CHECK(scripted.blocks_read >= 7 + 5);
}

TEST_CASE(candidates_at_the_boundary) {
    // q - 1 = 0x7FE000 is kept, q = 0x7FE001 is not, and the top bit of the third byte is
    // ignored, so 0xFFE000 is q - 1 again. Zero bytes give zero coefficients.
    std::array<std::uint8_t, ppq::SHAKE128_RATE> block{};
    const std::uint8_t lead[] = {0x00, 0xE0, 0x7F, 0x01, 0xE0, 0x7F,
                                 0x00, 0xE0, 0xFF, 0x05, 0x00, 0x00};
    std::copy(std::begin(lead), std::end(lead), block.begin());

    ScriptedBlocks scripted;
    scripted.script.assign(5, block);
    Poly a{};
    ppq::rej_ntt_poly(scripted, a.data());
    CHECK(a[0] == ppq::Q - 1);
    CHECK(a[1] == ppq::Q - 1);
    CHECK(a[2] == 5);
    CHECK(a[3] == 0);
    // 55 candidates per block, one rejected: 55 coefficients each
    CHECK(a[55] == ppq::Q - 1 && a[56] == ppq::Q - 1 && a[57] == 5);
    CHECK(scripted.blocks_read == 5);
}

} // namespace

int main() { return check::run_all(); }
