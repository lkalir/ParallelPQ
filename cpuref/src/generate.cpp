#include "ppq/generate.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

#include "ppq/drbg.h"
#include "ppq/parallel.h"

namespace ppq {
namespace {

using cpuref::Bytes;

/// What each signature will look like, decided serially so the parallel phase only fills
/// pre-sized slots
struct Plan {
    std::vector<Tamper> tamper;
    std::vector<std::size_t> msg_len;
    std::vector<std::size_t> ctx_len;
    /// Message tampers on an empty message: sign the empty message, then store one byte
    std::vector<std::uint8_t> grown;
};

Plan make_plan(const GenSpec &spec) {
    const std::size_t n = spec.signatures;
    Plan plan;
    plan.tamper.assign(n, Tamper::None);
    plan.msg_len.resize(n);
    plan.ctx_len.resize(n);
    plan.grown.assign(n, 0);

    // Exactly round(n * (1 - pass_rate)) failures at positions from a seeded shuffle
    const auto passing =
        static_cast<std::size_t>(std::llround(static_cast<double>(n) * spec.pass_rate));
    const std::size_t failing = n - std::min(passing, n);

    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    Drbg shuffle(spec.seed, "fail-order");
    for (std::size_t i = n; i > 1; --i) {
        std::swap(order[i - 1], order[shuffle.uniform(i)]);
    }
    for (std::size_t k = 0; k < failing; ++k) {
        plan.tamper[order[k]] = spec.fail_modes[k % spec.fail_modes.size()];
    }

    Drbg lengths(spec.seed, "lengths");
    for (std::size_t i = 0; i < n; ++i) {
        plan.msg_len[i] = lengths.range(spec.msg_min, spec.msg_max);
        plan.ctx_len[i] = lengths.range(0, spec.ctx_max);
        // A message flip needs a byte to flip: an empty message gains one after signing
        if (plan.tamper[i] == Tamper::Message && plan.msg_len[i] == 0) {
            plan.msg_len[i] = 1;
            plan.grown[i] = 1;
        }
    }
    return plan;
}

/// Sets one z coefficient to gamma1 (packed as gamma1 - z = 0), breaking the norm bound.
/// Coefficients are packed little-endian in z_bits; group starts are byte aligned (4
/// coefficients per 9 bytes at 18 bits, 2 per 5 bytes at 20 bits).
void tamper_z(const ParamSet &p, std::uint8_t *sig, Drbg &rng) {
    const std::size_t per_group = p.z_bits == 18 ? 4 : 2;
    const std::size_t group_bytes = per_group * static_cast<std::size_t>(p.z_bits) / 8;
    const std::size_t groups = N / per_group;

    const std::size_t poly = rng.uniform(static_cast<std::uint64_t>(p.l));
    std::uint8_t *c =
        sig + p.ctilde_bytes + poly * p.z_poly_bytes + rng.uniform(groups) * group_bytes;

    c[0] = 0;
    c[1] = 0;
    c[2] &= static_cast<std::uint8_t>(0xFF << (p.z_bits - 16));
}

/// Makes HintBitUnpack fail: either a cumulative count above omega, or a nonzero byte in
/// the unused index padding (when there is any)
void tamper_hint(const ParamSet &p, std::uint8_t *sig, Drbg &rng) {
    std::uint8_t *h = sig + p.sig_bytes - p.hint_bytes;
    const auto omega = static_cast<std::size_t>(p.omega);
    const std::size_t used = h[omega + static_cast<std::size_t>(p.k) - 1];

    if (used < omega && rng.uniform(2) == 0) {
        h[omega - 1] = 1;
    } else {
        h[omega + static_cast<std::size_t>(p.k) - 1] = static_cast<std::uint8_t>(omega + 1);
    }
}

} // namespace

void validate_spec(const GenSpec &spec) {
    const auto bad = [](const std::string &what) {
        throw std::invalid_argument("bad workload spec: " + what);
    };

    if (find_param_set(spec.params.id) == nullptr) {
        bad("unknown parameter set");
    }
    if (spec.keys > spec.signatures && spec.signatures > 0) {
        bad("more keys than signatures");
    }
    if (spec.keys > 0xFFFFFFFFu) {
        bad("too many keys");
    }
    if (!(spec.pass_rate >= 0.0 && spec.pass_rate <= 1.0)) {
        bad("pass rate must be in [0, 1]");
    }
    if (spec.msg_min > spec.msg_max) {
        bad("minimum message length exceeds the maximum");
    }
    if (spec.ctx_max > MAX_CONTEXT_BYTES) {
        bad("contexts are at most 255 bytes");
    }
    if (spec.pass_rate < 1.0 && spec.fail_modes.empty()) {
        bad("failing signatures need at least one tamper mode");
    }
    for (const Tamper t : spec.fail_modes) {
        if (t == Tamper::None || static_cast<std::size_t>(t) >= TAMPER_COUNT) {
            bad("tamper modes must be message, commitment, z-norm or hint");
        }
    }
    if (!cpuref::available(spec.impl)) {
        bad(std::string("CPU implementation '") + cpuref::impl_name(spec.impl) +
            "' is not available");
    }
}

Workload generate(const GenSpec &spec) {
    validate_spec(spec);

    const ParamSet &p = spec.params;
    const std::size_t n = spec.signatures;
    const std::size_t keys = n == 0 ? 0 : (spec.keys == 0 ? n : spec.keys);
    const Plan plan = make_plan(spec);

    Workload w;
    w.params = *find_param_set(p.id);
    w.seed = spec.seed;
    w.pks.resize(keys * p.pk_bytes);
    w.sigs.resize(n * p.sig_bytes);
    w.key_index.resize(n);
    w.expected.resize(n);
    w.tamper = plan.tamper;

    w.msg_offsets.assign(n + 1, 0);
    w.ctx_offsets.assign(n + 1, 0);
    for (std::size_t i = 0; i < n; ++i) {
        w.msg_offsets[i + 1] = w.msg_offsets[i] + plan.msg_len[i];
        w.ctx_offsets[i + 1] = w.ctx_offsets[i] + plan.ctx_len[i];
        w.key_index[i] = static_cast<std::uint32_t>(i % keys);
        w.expected[i] = expected_verdict(plan.tamper[i]);
    }
    w.msg_data.resize(w.msg_offsets[n]);
    w.ctx_data.resize(w.ctx_offsets[n]);

    // Secret keys are only needed while signing
    std::vector<Bytes> sks(keys);

    parallel_for(
        keys, spec.threads,
        [&](std::size_t k) {
            Bytes xi(SEED_BYTES);
            Drbg(spec.seed, "key", k).fill(xi.data(), xi.size());
            cpuref::KeyPair kp = cpuref::keygen(p, xi, spec.impl);
            std::copy(kp.pk.begin(), kp.pk.end(),
                      w.pks.begin() + static_cast<std::ptrdiff_t>(k * p.pk_bytes));
            sks[k] = std::move(kp.sk);
        },
        4);

    parallel_for(
        n, spec.threads,
        [&](std::size_t i) {
            Drbg rng(spec.seed, "signature", i);

            std::uint8_t *msg = w.msg_data.data() + w.msg_offsets[i];
            std::uint8_t *ctx = w.ctx_data.data() + w.ctx_offsets[i];
            const std::size_t msg_len = plan.msg_len[i];
            const std::size_t ctx_len = plan.ctx_len[i];
            rng.fill(msg, msg_len);
            rng.fill(ctx, ctx_len);

            Bytes rnd(RND_BYTES, 0);
            if (!spec.deterministic) {
                rng.fill(rnd.data(), rnd.size());
            }

            const std::size_t signed_len = plan.grown[i] != 0 ? 0 : msg_len;

            const Bytes sig = cpuref::sign(p, sks[w.key_index[i]], ByteView(msg, signed_len),
                                           ByteView(ctx, ctx_len), rnd, spec.impl);
            std::uint8_t *out = w.sigs.data() + i * p.sig_bytes;
            std::copy(sig.begin(), sig.end(), out);

            switch (plan.tamper[i]) {
            case Tamper::None:
                break;
            case Tamper::Message:
                msg[rng.uniform(msg_len)] ^= static_cast<std::uint8_t>(1u << rng.uniform(8));
                break;
            case Tamper::Commitment:
                out[rng.uniform(p.ctilde_bytes)] ^= static_cast<std::uint8_t>(1u << rng.uniform(8));
                break;
            case Tamper::ZNorm:
                tamper_z(p, out, rng);
                break;
            case Tamper::Hint:
                tamper_hint(p, out, rng);
                break;
            }
        },
        4);

    w.validate();
    return w;
}

std::vector<std::uint8_t> cpu_verify_batch(const Workload &w, cpuref::Impl impl, unsigned threads) {
    std::vector<std::uint8_t> ok(w.sig_count(), 0);
    if (!cpuref::supports(impl, w.iface)) {
        throw std::invalid_argument(std::string("CPU implementation '") + cpuref::impl_name(impl) +
                                    "' cannot verify " + interface_name(w.iface) + " workloads");
    }
    parallel_for(w.sig_count(), threads, [&](std::size_t i) {
        bool valid = false;
        switch (w.iface) {
        case Interface::External:
            valid = cpuref::verify(w.params, w.pk_of(i), w.msg(i), w.ctx(i), w.sig(i), impl);
            break;
        case Interface::Internal:
            valid = cpuref::verify_internal(w.params, w.pk_of(i), w.msg(i), w.sig(i), impl);
            break;
        case Interface::Mu:
            valid = cpuref::verify_mu(w.params, w.pk_of(i), w.msg(i), w.sig(i), impl);
            break;
        }
        ok[i] = valid ? 1 : 0;
    });
    return ok;
}

} // namespace ppq
