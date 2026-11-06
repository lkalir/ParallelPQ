/// CPU reference wrapper: determinism, round trips, rejections, and agreement between the
/// pq-crystals ref, pq-crystals AVX2 and OpenSSL implementations.

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "ppq/cpuref.h"
#include "ppq/cpuref_stages.h"
#include "ppq/drbg.h"

namespace {

using ppq::cpuref::Bytes;
using ppq::cpuref::Impl;

constexpr std::uint64_t TEST_SEED = 0x7070715f63707572; // "ppq_cpur"
constexpr std::size_t LONG_MESSAGE_BYTES = 5000;

Bytes random_bytes(std::uint64_t index, std::size_t n, const char *label = "test") {
    Bytes out(n);
    ppq::Drbg(TEST_SEED, label, index).fill(out.data(), n);
    return out;
}

Bytes seed_for(std::uint64_t index) { return random_bytes(index, ppq::SEED_BYTES, "xi"); }

template <class Fn> bool throws_invalid(Fn &&fn) {
    try {
        fn();
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

TEST_CASE(reports_available_implementations) {
    CHECK(ppq::cpuref::available(Impl::Ref));
    std::cout << "    available:";
    for (const Impl impl : ppq::cpuref::available_impls()) {
        std::cout << " " << ppq::cpuref::impl_name(impl);
    }
    std::cout << "\n";
}

TEST_CASE(keygen_is_a_function_of_the_seed) {
    for (const ppq::ParamSet &p : ppq::PARAM_SETS) {
        const auto a = ppq::cpuref::keygen(p, seed_for(1));
        const auto b = ppq::cpuref::keygen(p, seed_for(1));
        const auto c = ppq::cpuref::keygen(p, seed_for(2));

        CHECK(a.pk.size() == p.pk_bytes);
        CHECK(a.sk.size() == p.sk_bytes);
        CHECK(a.pk == b.pk && a.sk == b.sk);
        CHECK(a.pk != c.pk);
        // pk = rho || t1 and sk = rho || K || tr || ...: both start with rho
        CHECK(std::equal(a.pk.begin(), a.pk.begin() + ppq::SEED_BYTES, a.sk.begin()));
    }
}

TEST_CASE(sign_then_verify_round_trips) {
    const std::vector<std::pair<Bytes, Bytes>> cases = {
        {{}, {}},                                                  // empty message and context
        {random_bytes(1, 33), {}},                                 // short message
        {random_bytes(2, LONG_MESSAGE_BYTES), random_bytes(3, 7)}, // long message with context
        {random_bytes(4, 1), random_bytes(5, ppq::MAX_CONTEXT_BYTES)},
    };

    for (const ppq::ParamSet &p : ppq::PARAM_SETS) {
        const auto kp = ppq::cpuref::keygen(p, seed_for(10));

        for (std::size_t i = 0; i < cases.size(); ++i) {
            const auto &[msg, ctx] = cases[i];
            const Bytes sig = ppq::cpuref::sign(p, kp.sk, msg, ctx, random_bytes(i, 32, "rnd"));
            CHECK(sig.size() == p.sig_bytes);
            CHECK(ppq::cpuref::verify(p, kp.pk, msg, ctx, sig));
        }
    }
}

TEST_CASE(signing_is_a_function_of_rnd) {
    const ppq::ParamSet &p = ppq::ML_DSA_65;
    const auto kp = ppq::cpuref::keygen(p, seed_for(20));
    const Bytes msg = random_bytes(20, 64);
    const Bytes zero(ppq::RND_BYTES, 0);

    const Bytes d1 = ppq::cpuref::sign(p, kp.sk, msg, {}, zero);
    const Bytes d2 = ppq::cpuref::sign(p, kp.sk, msg, {}, zero);
    const Bytes h = ppq::cpuref::sign(p, kp.sk, msg, {}, random_bytes(21, 32, "rnd"));

    CHECK(d1 == d2);
    CHECK(d1 != h);
    CHECK(ppq::cpuref::verify(p, kp.pk, msg, {}, h));
}

TEST_CASE(verify_rejects_tampering) {
    for (const ppq::ParamSet &p : ppq::PARAM_SETS) {
        const auto kp = ppq::cpuref::keygen(p, seed_for(30));
        const auto other = ppq::cpuref::keygen(p, seed_for(31));
        const Bytes msg = random_bytes(30, 100);
        const Bytes ctx = random_bytes(31, 10);
        const Bytes sig = ppq::cpuref::sign(p, kp.sk, msg, ctx, random_bytes(30, 32, "rnd"));

        Bytes bad_msg = msg;
        bad_msg[50] ^= 0x01;
        Bytes bad_ctx = ctx;
        bad_ctx[0] ^= 0x80;
        Bytes bad_ctilde = sig;
        bad_ctilde[0] ^= 0x01;
        Bytes bad_z = sig;
        bad_z[p.ctilde_bytes + 10] ^= 0x10;
        Bytes short_sig(sig.begin(), sig.end() - 1);
        Bytes long_ctx(ppq::MAX_CONTEXT_BYTES + 1, 0);

        CHECK(ppq::cpuref::verify(p, kp.pk, msg, ctx, sig));
        CHECK(!ppq::cpuref::verify(p, kp.pk, bad_msg, ctx, sig));
        CHECK(!ppq::cpuref::verify(p, kp.pk, msg, bad_ctx, sig));
        CHECK(!ppq::cpuref::verify(p, kp.pk, msg, {}, sig));
        CHECK(!ppq::cpuref::verify(p, kp.pk, msg, ctx, bad_ctilde));
        CHECK(!ppq::cpuref::verify(p, kp.pk, msg, ctx, bad_z));
        CHECK(!ppq::cpuref::verify(p, other.pk, msg, ctx, sig));
        CHECK(!ppq::cpuref::verify(p, kp.pk, msg, ctx, short_sig));
        CHECK(!ppq::cpuref::verify(p, kp.pk, msg, long_ctx, sig));
    }
}

TEST_CASE(bad_arguments_throw) {
    const ppq::ParamSet &p = ppq::ML_DSA_44;
    const auto kp = ppq::cpuref::keygen(p, seed_for(40));
    const Bytes rnd(ppq::RND_BYTES, 0);

    CHECK(throws_invalid([&] { ppq::cpuref::keygen(p, Bytes(31, 0)); }));
    CHECK(throws_invalid([&] { ppq::cpuref::sign(p, kp.sk, {}, {}, Bytes(31, 0)); }));
    CHECK(throws_invalid([&] { ppq::cpuref::sign(p, kp.pk, {}, {}, rnd); }));
    CHECK(throws_invalid(
        [&] { ppq::cpuref::sign(p, kp.sk, {}, Bytes(ppq::MAX_CONTEXT_BYTES + 1, 0), rnd); }));
    CHECK(throws_invalid([&] { ppq::cpuref::keygen(ppq::ParamSet{}, seed_for(0)); }));

    for (const Impl impl : ppq::cpuref::ALL_IMPLS) {
        if (!ppq::cpuref::available(impl)) {
            CHECK(throws_invalid([&] { ppq::cpuref::keygen(p, seed_for(0), impl); }));
        }
    }
}

/// KeyGen_internal and Sign_internal are deterministic given xi and rnd, so independent
/// implementations must produce identical bytes, and each must verify the others' output.
TEST_CASE(implementations_agree_byte_for_byte) {
    const auto impls = ppq::cpuref::available_impls();
    if (impls.size() < 2) {
        throw check::Skip("only one CPU implementation available");
    }

    for (const ppq::ParamSet &p : ppq::PARAM_SETS) {
        const Bytes xi = seed_for(50 + static_cast<std::uint64_t>(p.id));
        const Bytes msg = random_bytes(50, 300);
        const Bytes ctx = random_bytes(51, 12);
        const Bytes rnd = random_bytes(52, 32, "rnd");

        const auto ref_kp = ppq::cpuref::keygen(p, xi, Impl::Ref);
        const Bytes ref_sig = ppq::cpuref::sign(p, ref_kp.sk, msg, ctx, rnd, Impl::Ref);

        for (const Impl impl : impls) {
            const std::string what = std::string(p.name) + " " + ppq::cpuref::impl_name(impl);
            const auto kp = ppq::cpuref::keygen(p, xi, impl);
            CHECK_EQ_BYTES(kp.pk, ref_kp.pk, what + " pk");
            CHECK_EQ_BYTES(kp.sk, ref_kp.sk, what + " sk");

            const Bytes sig = ppq::cpuref::sign(p, ref_kp.sk, msg, ctx, rnd, impl);
            CHECK_EQ_BYTES(sig, ref_sig, what + " signature");

            for (const Impl verifier : impls) {
                CHECK(ppq::cpuref::verify(p, ref_kp.pk, msg, ctx, sig, verifier));
                Bytes bad = sig;
                bad[bad.size() / 2] ^= 0x04;
                CHECK(!ppq::cpuref::verify(p, ref_kp.pk, msg, ctx, bad, verifier));
            }
        }
    }
}

/// Verify, Verify_internal on M' = 0 || |ctx| || ctx || M, and verification from mu are three
/// views of one check; every implementation must agree on all of them.
TEST_CASE(internal_and_mu_interfaces_agree_with_verify) {
    for (const ppq::ParamSet &p : ppq::PARAM_SETS) {
        const auto kp = ppq::cpuref::keygen(p, seed_for(60));
        const Bytes msg = random_bytes(60, 77);
        const Bytes ctx = random_bytes(61, 9);
        const Bytes sig = ppq::cpuref::sign(p, kp.sk, msg, ctx, random_bytes(62, 32, "rnd"));
        Bytes bad = sig;
        bad[3] ^= 0x20;

        Bytes mprime = {0, static_cast<std::uint8_t>(ctx.size())};
        mprime.insert(mprime.end(), ctx.begin(), ctx.end());
        mprime.insert(mprime.end(), msg.begin(), msg.end());
        const Bytes mu = ppq::cpuref::mu_external(p, kp.pk, msg, ctx);
        CHECK(mu == ppq::cpuref::mu_internal(p, kp.pk, mprime));

        for (const Impl impl : ppq::cpuref::available_impls()) {
            CHECK(ppq::cpuref::verify_internal(p, kp.pk, mprime, sig, impl));
            CHECK(!ppq::cpuref::verify_internal(p, kp.pk, msg, sig, impl));
            CHECK(!ppq::cpuref::verify_internal(p, kp.pk, mprime, bad, impl));
            if (ppq::cpuref::supports(impl, ppq::cpuref::Interface::Mu)) {
                CHECK(ppq::cpuref::verify_mu(p, kp.pk, mu, sig, impl));
                CHECK(!ppq::cpuref::verify_mu(p, kp.pk, mu, bad, impl));
            } else {
                CHECK(throws_invalid([&] { ppq::cpuref::verify_mu(p, kp.pk, mu, sig, impl); }));
            }
        }
    }
}

TEST_CASE(stage_verdicts_follow_the_tampering) {
    const ppq::ParamSet &p = ppq::ML_DSA_44;
    const auto kp = ppq::cpuref::keygen(p, seed_for(70));
    const Bytes msg = random_bytes(70, 40);
    const Bytes sig = ppq::cpuref::sign(p, kp.sk, msg, {}, random_bytes(71, 32, "rnd"));
    const Bytes mu = ppq::cpuref::mu_external(p, kp.pk, msg, {});

    const auto good = ppq::cpuref::sig_stages(p, kp.pk, mu, sig);
    CHECK(good.verdict == ppq::Verdict::Pass);
    CHECK(good.hint_ok);
    CHECK(good.ctilde == good.ctilde_prime);
    CHECK(good.z.size() == 4 && good.w1.size() == 4 && good.w1_packed.size() == p.w1_bytes);

    Bytes bad_mu = mu;
    bad_mu[0] ^= 1;
    CHECK(ppq::cpuref::sig_stages(p, kp.pk, bad_mu, sig).verdict == ppq::Verdict::BadCommitment);

    Bytes bad_hint = sig;
    bad_hint.back() = static_cast<std::uint8_t>(p.omega + 1);
    const auto hint = ppq::cpuref::sig_stages(p, kp.pk, mu, bad_hint);
    CHECK(hint.verdict == ppq::Verdict::BadHint);
    CHECK(hint.h.empty() && hint.w1.empty() && hint.ctilde_prime.empty());
    CHECK(hint.z_hat == good.z_hat);

    Bytes big_z = sig;
    big_z[p.ctilde_bytes] = 0; // first z coefficient packed as 0 -> z = gamma1
    big_z[p.ctilde_bytes + 1] = 0;
    big_z[p.ctilde_bytes + 2] &= 0xFC;
    CHECK(ppq::cpuref::sig_stages(p, kp.pk, mu, big_z).verdict == ppq::Verdict::ZNorm);

    const auto ks = ppq::cpuref::key_stages(p, kp.pk);
    CHECK(ks.a_hat.size() == 16 && ks.t1_hat.size() == 4);
    CHECK(std::equal(ks.rho.begin(), ks.rho.end(), kp.pk.begin()));
}

TEST_CASE(ntt_round_trips) {
    ppq::cpuref::Poly a{};
    ppq::Drbg rng(TEST_SEED, "ntt");
    for (auto &c : a) {
        c = static_cast<std::int32_t>(rng.uniform(ppq::Q));
    }
    const auto a_hat = ppq::cpuref::ntt(a);
    CHECK(a_hat != a);
    CHECK(ppq::cpuref::intt(a_hat) == a);
    for (const std::int32_t c : a_hat) {
        CHECK(c >= 0 && c < ppq::Q);
    }
}

} // namespace

int main() { return check::run_all(); }
