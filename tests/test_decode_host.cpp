/// Key and signature decoding (ppq/decode.h) on the host: the stage and ACVP vectors, and
/// fuzzing against the naive FIPS 204 transcription

#include <algorithm>

#include "check.h"
#include "naive_mldsa.h"
#include "ppq/acvp.h"
#include "ppq/decode.h"
#include "ppq/drbg.h"
#include "vector_util.h"

namespace {

using naive::Bytes;
using naive::Poly;
using vector_util::params_of;
using vector_util::polys;

constexpr std::uint64_t TEST_SEED = 0x6465636f6465; // "decode"

const std::vector<ppq::KvRecord> &stages() {
    static const auto r = ppq::load_kv_records(TEST_VECTORS_DIR "/stages/mldsa_stages.txt");
    return r;
}

struct Decoded {
    std::vector<Poly> z;
    bool z_ok = true;
    bool hint_ok = false;
    std::vector<Poly> h;
};

/// ppq's decoders over a whole signature
Decoded decode(const ppq::ParamSet &p, const Bytes &sig) {
    return ppq::with_params(p.id, [&](auto m) {
        using M = decltype(m);
        Decoded d;
        d.z.resize(M::l);
        for (int j = 0; j < M::l; ++j) {
            ppq::z_decode_poly<M::id>(sig.data() + M::ctilde_bytes + j * M::z_poly_bytes,
                                      d.z[j].data());
            d.z_ok = d.z_ok && ppq::z_poly_norm_ok<M::id>(d.z[j].data());
        }
        std::vector<std::uint8_t> h(M::k * ppq::N);
        d.hint_ok = ppq::hint_decode<M::id>(sig.data() + M::sig_bytes - M::hint_bytes, h.data());
        if (d.hint_ok) {
            d.h.resize(M::k);
            for (int i = 0; i < M::k; ++i) {
                std::copy(h.begin() + i * ppq::N, h.begin() + (i + 1) * ppq::N, d.h[i].begin());
            }
        }
        return d;
    });
}

TEST_CASE(stage_vectors_decode) {
    for (const ppq::KvRecord &r : stages()) {
        if (r.str("kind") == "ntt") {
            continue;
        }
        const ppq::ParamSet &p = params_of(r);
        if (r.str("kind") == "key") {
            const Bytes pk = r.bytes("pk");
            const auto want = polys(r, "t1", static_cast<std::size_t>(p.k));
            for (int i = 0; i < p.k; ++i) {
                Poly t1{};
                ppq::t1_decode_poly(pk.data() + ppq::SEED_BYTES + i * ppq::T1_PACKED_BYTES,
                                    t1.data());
                CHECK(t1 == want[static_cast<std::size_t>(i)]);
            }
        } else if (r.str("kind") == "sig") {
            const Decoded d = decode(p, r.bytes("sig"));
            const ppq::Verdict v = vector_util::parse_verdict(r);
            CHECK(d.z == polys(r, "z", static_cast<std::size_t>(p.l)));
            CHECK(d.hint_ok == (r.integer("hint_ok") == 1));
            CHECK(d.hint_ok == (v != ppq::Verdict::BadHint));
            if (d.hint_ok) {
                CHECK(d.h == polys(r, "h", static_cast<std::size_t>(p.k)));
                CHECK(d.z_ok == (v != ppq::Verdict::ZNorm));
            }
        }
    }
}

TEST_CASE(acvp_hints_and_norms) {
    const auto groups = ppq::load_acvp_sigver(TEST_VECTORS_DIR "/acvp/mldsa_sigver.txt");
    std::size_t bad_hints = 0;
    for (const ppq::AcvpWorkload &g : groups) {
        const ppq::Workload &w = g.workload;
        for (std::size_t i = 0; i < w.sig_count(); ++i) {
            const Bytes sig(w.sig(i).begin(), w.sig(i).end());
            const Decoded d = decode(w.params, sig);
            CHECK(d.hint_ok == (w.expected[i] != ppq::Verdict::BadHint));
            CHECK(d.hint_ok == naive::hint_decode(w.params, sig).has_value());
            CHECK(d.z == naive::z_decode(w.params, sig));
            bad_hints += !d.hint_ok;
            // ACVP's z modifications stay inside the norm bound
            CHECK(d.z_ok);
        }
    }
    CHECK(bad_hints == 27);
}

/// A random valid hint encoding: per polynomial, a sorted set of distinct indices, at most
/// omega in total, with zero padding
Bytes random_valid_hint(const ppq::ParamSet &p, ppq::Drbg &rng) {
    Bytes y(p.hint_bytes, 0);
    const auto omega = static_cast<std::size_t>(p.omega);
    std::size_t used = 0;
    for (int i = 0; i < p.k; ++i) {
        const std::size_t room = omega - used;
        const std::size_t n = rng.uniform(std::min<std::size_t>(room, 20) + 1);
        std::vector<std::uint8_t> idx;
        while (idx.size() < n) {
            const auto v = static_cast<std::uint8_t>(rng.uniform(256));
            if (std::find(idx.begin(), idx.end(), v) == idx.end()) {
                idx.push_back(v);
            }
        }
        std::sort(idx.begin(), idx.end());
        std::copy(idx.begin(), idx.end(), y.begin() + static_cast<std::ptrdiff_t>(used));
        used += n;
        y[omega + static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(used);
    }
    return y;
}

TEST_CASE(hint_decode_matches_naive_under_fuzzing) {
    std::size_t valid = 0;
    std::size_t invalid = 0;
    for (const ppq::ParamSet &p : ppq::PARAM_SETS) {
        ppq::Drbg rng(TEST_SEED, p.name);
        for (int trial = 0; trial < 6000; ++trial) {
            Bytes y = random_valid_hint(p, rng);
            switch (rng.uniform(4)) {
            case 0: // leave valid
                break;
            case 1: // corrupt one byte anywhere
                y[rng.uniform(y.size())] = static_cast<std::uint8_t>(rng.uniform(256));
                break;
            case 2: // corrupt a count
                y[static_cast<std::size_t>(p.omega) +
                  rng.uniform(static_cast<std::uint64_t>(p.k))] =
                    static_cast<std::uint8_t>(rng.uniform(static_cast<std::uint64_t>(p.omega) + 3));
                break;
            default: // noise
                rng.fill(y.data(), y.size());
            }

            Bytes sig(p.sig_bytes, 0);
            std::copy(y.begin(), y.end(), sig.end() - static_cast<std::ptrdiff_t>(y.size()));
            const Decoded d = decode(p, sig);
            const auto n = naive::hint_decode(p, sig);
            CHECK(d.hint_ok == n.has_value());
            if (d.hint_ok && n) {
                CHECK(d.h == *n);
            }
            (d.hint_ok ? valid : invalid) += 1;
        }
    }
    // Both outcomes are well represented
    CHECK(valid > 3000 && invalid > 3000);
}

TEST_CASE(z_decode_matches_naive_on_random_bytes) {
    for (const ppq::ParamSet &p : ppq::PARAM_SETS) {
        ppq::Drbg rng(TEST_SEED, "z", static_cast<std::uint64_t>(p.id));
        for (int trial = 0; trial < 50; ++trial) {
            Bytes sig(p.sig_bytes);
            rng.fill(sig.data(), sig.size());
            const Decoded d = decode(p, sig);
            const auto z = naive::z_decode(p, sig);
            CHECK(d.z == z);
            CHECK(d.z_ok == (naive::max_abs(z) < p.gamma1 - p.beta));
        }

        // The extremes of the packing: all-zero bytes decode to gamma1, all-one bits to
        // gamma1 - (2^bits - 1) = -gamma1 + 1
        Bytes zeros(p.sig_bytes, 0);
        Bytes ones(p.sig_bytes, 0xFF);
        const Decoded dz = decode(p, zeros);
        const Decoded d1 = decode(p, ones);
        CHECK(dz.z[0][0] == p.gamma1 && !dz.z_ok);
        CHECK(d1.z[0][0] == -p.gamma1 + 1 && !d1.z_ok);
    }
}

TEST_CASE(norm_bound_is_exclusive) {
    ppq::with_params(65, [](auto m) {
        using M = decltype(m);
        Poly z{};
        z[7] = M::gamma1 - M::beta - 1;
        CHECK(ppq::z_poly_norm_ok<M::id>(z.data()));
        z[7] = -(M::gamma1 - M::beta - 1);
        CHECK(ppq::z_poly_norm_ok<M::id>(z.data()));
        z[7] = M::gamma1 - M::beta;
        CHECK(!ppq::z_poly_norm_ok<M::id>(z.data()));
        z[7] = -(M::gamma1 - M::beta);
        CHECK(!ppq::z_poly_norm_ok<M::id>(z.data()));
        return 0;
    });
}

} // namespace

int main() { return check::run_all(); }
