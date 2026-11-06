#include "ppq/cpuref_stages.h"

#include <memory>
#include <stdexcept>
#include <string>

#include "pqc_stages.h"

namespace ppq::cpuref {
namespace {

struct ModeFns {
    void (*key_stages)(ppq_pqc_key_stages *, const std::uint8_t *);
    void (*sig_stages)(ppq_pqc_sig_stages *, const std::uint8_t *, const std::uint8_t *,
                       const std::uint8_t *);
    void (*mu)(std::uint8_t *, const std::uint8_t *, const std::uint8_t *, std::size_t,
               const std::uint8_t *, std::size_t);
};

const ModeFns &mode_fns(const ParamSet &params) {
    static const ModeFns FNS[] = {
        {ppq_pqc2_key_stages, ppq_pqc2_sig_stages, ppq_pqc2_mu},
        {ppq_pqc3_key_stages, ppq_pqc3_sig_stages, ppq_pqc3_mu},
        {ppq_pqc5_key_stages, ppq_pqc5_sig_stages, ppq_pqc5_mu},
    };
    switch (params.id) {
    case 44:
        return FNS[0];
    case 65:
        return FNS[1];
    case 87:
        return FNS[2];
    default:
        throw std::invalid_argument("unknown ML-DSA parameter set " + std::to_string(params.id));
    }
}

void require_size(ByteView v, std::size_t want, const char *what) {
    if (v.size() != want) {
        throw std::invalid_argument(std::string(what) + " must be " + std::to_string(want) +
                                    " bytes, got " + std::to_string(v.size()));
    }
}

Poly to_poly(const std::int32_t *p) {
    Poly out;
    std::copy(p, p + N, out.begin());
    return out;
}

std::vector<Poly> rows(const std::int32_t (*p)[PPQ_PQC_N], int count) {
    std::vector<Poly> out;
    for (int i = 0; i < count; ++i) {
        out.push_back(to_poly(p[i]));
    }
    return out;
}

const std::uint8_t *ptr(ByteView v) {
    static const std::uint8_t EMPTY = 0;
    return v.empty() ? &EMPTY : v.data();
}

} // namespace

KeyStages key_stages(const ParamSet &params, ByteView pk) {
    const ModeFns &fns = mode_fns(params);
    require_size(pk, params.pk_bytes, "public key");

    auto raw = std::make_unique<ppq_pqc_key_stages>();
    fns.key_stages(raw.get(), pk.data());

    KeyStages out;
    out.rho.assign(raw->rho, raw->rho + SEED_BYTES);
    out.t1 = rows(raw->t1, params.k);
    out.tr.assign(raw->tr, raw->tr + TR_BYTES);
    for (int i = 0; i < params.k; ++i) {
        for (int j = 0; j < params.l; ++j) {
            out.a_hat.push_back(to_poly(raw->a_hat[i][j]));
        }
    }
    out.t1_hat = rows(raw->t1_hat, params.k);
    return out;
}

SigStages sig_stages(const ParamSet &params, ByteView pk, ByteView mu, ByteView sig) {
    const ModeFns &fns = mode_fns(params);
    require_size(pk, params.pk_bytes, "public key");
    require_size(mu, MU_BYTES, "mu");
    require_size(sig, params.sig_bytes, "signature");

    auto raw = std::make_unique<ppq_pqc_sig_stages>();
    fns.sig_stages(raw.get(), pk.data(), mu.data(), sig.data());

    SigStages out;
    out.hint_ok = raw->hint_ok != 0;
    if (!out.hint_ok) {
        out.verdict = Verdict::BadHint;
    } else if (raw->znorm_ok == 0) {
        out.verdict = Verdict::ZNorm;
    } else {
        out.verdict = raw->commitment_ok != 0 ? Verdict::Pass : Verdict::BadCommitment;
    }

    out.ctilde.assign(raw->ctilde, raw->ctilde + params.ctilde_bytes);
    out.z = rows(raw->z, params.l);
    out.c = to_poly(raw->c);
    out.z_hat = rows(raw->z_hat, params.l);
    out.c_hat = to_poly(raw->c_hat);

    if (out.hint_ok) {
        out.h = rows(raw->h, params.k);
        out.w_approx = rows(raw->w_approx, params.k);
        out.w1 = rows(raw->w1, params.k);
        out.w1_packed.assign(raw->w1_packed, raw->w1_packed + params.w1_bytes);
        out.ctilde_prime.assign(raw->ctilde_prime, raw->ctilde_prime + params.ctilde_bytes);
    }
    return out;
}

Bytes mu_external(const ParamSet &params, ByteView pk, ByteView msg, ByteView ctx) {
    const ModeFns &fns = mode_fns(params);
    require_size(pk, params.pk_bytes, "public key");
    if (ctx.size() > MAX_CONTEXT_BYTES) {
        throw std::invalid_argument("context strings are at most 255 bytes");
    }

    std::uint8_t pre[2 + MAX_CONTEXT_BYTES] = {0, static_cast<std::uint8_t>(ctx.size())};
    std::copy(ctx.begin(), ctx.end(), pre + 2);

    Bytes mu(MU_BYTES);
    fns.mu(mu.data(), pk.data(), pre, 2 + ctx.size(), ptr(msg), msg.size());
    return mu;
}

Bytes mu_internal(const ParamSet &params, ByteView pk, ByteView mprime) {
    const ModeFns &fns = mode_fns(params);
    require_size(pk, params.pk_bytes, "public key");

    Bytes mu(MU_BYTES);
    fns.mu(mu.data(), pk.data(), ptr({}), 0, ptr(mprime), mprime.size());
    return mu;
}

// The NTT does not depend on the parameter set; any mode's build will do
Poly ntt(const Poly &a) {
    Poly out = a;
    ppq_pqc2_ntt(out.data());
    return out;
}

Poly intt(const Poly &a_hat) {
    Poly out = a_hat;
    ppq_pqc2_intt(out.data());
    return out;
}

} // namespace ppq::cpuref
