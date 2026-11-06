#include "ppq/cpuref.h"

#include <stdexcept>
#include <string>

#include "openssl_impl.h"
#include "ppq/cpuref_stages.h"
#include "random_source.h"

// pq-crystals entry points. Each mode (2, 3, 5 = ML-DSA-44, -65, -87) and variant
// (ref, avx2) is compiled separately with its own symbol prefix, so all of them link
// into one binary. Declared here rather than through the library headers, whose
// macros (K, L, N, ...) would collide across modes.
#define PPQ_PQC_DECLARE(ns)                                                                        \
    extern "C" int ns##_keypair(std::uint8_t *pk, std::uint8_t *sk);                               \
    extern "C" int ns##_signature_internal(std::uint8_t *sig, std::size_t *siglen,                 \
                                           const std::uint8_t *m, std::size_t mlen,                \
                                           const std::uint8_t *pre, std::size_t prelen,            \
                                           const std::uint8_t *rnd, const std::uint8_t *sk);       \
    extern "C" int ns##_verify(const std::uint8_t *sig, std::size_t siglen, const std::uint8_t *m, \
                               std::size_t mlen, const std::uint8_t *ctx, std::size_t ctxlen,      \
                               const std::uint8_t *pk);                                            \
    extern "C" int ns##_verify_internal(                                                           \
        const std::uint8_t *sig, std::size_t siglen, const std::uint8_t *m, std::size_t mlen,      \
        const std::uint8_t *pre, std::size_t prelen, const std::uint8_t *pk);

PPQ_PQC_DECLARE(pqcrystals_dilithium2_ref)
PPQ_PQC_DECLARE(pqcrystals_dilithium3_ref)
PPQ_PQC_DECLARE(pqcrystals_dilithium5_ref)
#if PPQ_CPUREF_HAVE_AVX2
PPQ_PQC_DECLARE(pqcrystals_dilithium2_avx2)
PPQ_PQC_DECLARE(pqcrystals_dilithium3_avx2)
PPQ_PQC_DECLARE(pqcrystals_dilithium5_avx2)
#endif

namespace ppq::cpuref {
namespace {

struct PqcFns {
    int (*keypair)(std::uint8_t *, std::uint8_t *);
    int (*signature_internal)(std::uint8_t *, std::size_t *, const std::uint8_t *, std::size_t,
                              const std::uint8_t *, std::size_t, const std::uint8_t *,
                              const std::uint8_t *);
    int (*verify)(const std::uint8_t *, std::size_t, const std::uint8_t *, std::size_t,
                  const std::uint8_t *, std::size_t, const std::uint8_t *);
    int (*verify_internal)(const std::uint8_t *, std::size_t, const std::uint8_t *, std::size_t,
                           const std::uint8_t *, std::size_t, const std::uint8_t *);
};

#define PPQ_PQC_FNS(ns)                                                                            \
    PqcFns { ns##_keypair, ns##_signature_internal, ns##_verify, ns##_verify_internal }

const PqcFns *pqc_fns(Impl impl, int id) {
    static const PqcFns REF[] = {PPQ_PQC_FNS(pqcrystals_dilithium2_ref),
                                 PPQ_PQC_FNS(pqcrystals_dilithium3_ref),
                                 PPQ_PQC_FNS(pqcrystals_dilithium5_ref)};
#if PPQ_CPUREF_HAVE_AVX2
    static const PqcFns AVX2[] = {PPQ_PQC_FNS(pqcrystals_dilithium2_avx2),
                                  PPQ_PQC_FNS(pqcrystals_dilithium3_avx2),
                                  PPQ_PQC_FNS(pqcrystals_dilithium5_avx2)};
#endif

    const int index = id == 44 ? 0 : id == 65 ? 1 : id == 87 ? 2 : -1;
    if (index < 0) {
        return nullptr;
    }

    switch (impl) {
    case Impl::Ref:
        return &REF[index];
    case Impl::Avx2:
#if PPQ_CPUREF_HAVE_AVX2
        return &AVX2[index];
#else
        return nullptr;
#endif
    case Impl::OpenSsl:
        return nullptr;
    }
    return nullptr;
}

/// Non-null even for empty input; the C code is handed a pointer either way
const std::uint8_t *ptr(ByteView v) {
    static const std::uint8_t EMPTY = 0;
    return v.empty() ? &EMPTY : v.data();
}

void require_available(Impl impl) {
    if (!available(impl)) {
        throw std::invalid_argument(std::string("CPU implementation '") + impl_name(impl) +
                                    "' is not available in this build or on this CPU");
    }
}

void require_size(ByteView v, std::size_t want, const char *what) {
    if (v.size() != want) {
        throw std::invalid_argument(std::string(what) + " must be " + std::to_string(want) +
                                    " bytes, got " + std::to_string(v.size()));
    }
}

void require_known(const ParamSet &params) {
    if (find_param_set(params.id) == nullptr) {
        throw std::invalid_argument("unknown ML-DSA parameter set " + std::to_string(params.id));
    }
}

} // namespace

const char *impl_name(Impl impl) {
    switch (impl) {
    case Impl::Ref:
        return "ref";
    case Impl::Avx2:
        return "avx2";
    case Impl::OpenSsl:
        return "openssl";
    }
    return "unknown";
}

bool available(Impl impl) {
    switch (impl) {
    case Impl::Ref:
        return true;
    case Impl::Avx2:
#if PPQ_CPUREF_HAVE_AVX2
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("popcnt");
#else
        return false;
#endif
    case Impl::OpenSsl:
        return detail::openssl_available();
    }
    return false;
}

const char *interface_name(Interface iface) {
    switch (iface) {
    case Interface::External:
        return "external";
    case Interface::Internal:
        return "internal";
    case Interface::Mu:
        return "mu";
    }
    return "unknown";
}

bool supports(Impl impl, Interface iface) {
    // pq-crystals exposes no verify-from-mu entry; Ref gets one through its stage internals
    return available(impl) && !(impl == Impl::Avx2 && iface == Interface::Mu);
}

std::vector<Impl> available_impls() {
    std::vector<Impl> out;
    for (const Impl impl : ALL_IMPLS) {
        if (available(impl)) {
            out.push_back(impl);
        }
    }
    return out;
}

KeyPair keygen(const ParamSet &params, ByteView xi, Impl impl) {
    require_known(params);
    require_available(impl);
    require_size(xi, SEED_BYTES, "keygen seed xi");

    if (impl == Impl::OpenSsl) {
        return detail::openssl_keygen(params, xi);
    }

    KeyPair kp{Bytes(params.pk_bytes), Bytes(params.sk_bytes)};
    detail::ScopedRandomBytes seed(xi);
    pqc_fns(impl, params.id)->keypair(kp.pk.data(), kp.sk.data());

    if (seed.left() != 0) {
        throw std::logic_error("pq-crystals keypair drew fewer seed bytes than expected");
    }
    return kp;
}

Bytes sign(const ParamSet &params, ByteView sk, ByteView msg, ByteView ctx, ByteView rnd,
           Impl impl) {
    require_known(params);
    require_available(impl);
    require_size(sk, params.sk_bytes, "secret key");
    require_size(rnd, RND_BYTES, "signing randomness");
    if (ctx.size() > MAX_CONTEXT_BYTES) {
        throw std::invalid_argument("context strings are at most 255 bytes");
    }

    if (impl == Impl::OpenSsl) {
        return detail::openssl_sign(params, sk, msg, ctx, rnd);
    }

    // M' prefix for pure ML-DSA: 0 || len(ctx) || ctx (FIPS 204 Algorithm 2)
    std::uint8_t pre[2 + MAX_CONTEXT_BYTES];
    pre[0] = 0;
    pre[1] = static_cast<std::uint8_t>(ctx.size());
    for (std::size_t i = 0; i < ctx.size(); ++i) {
        pre[2 + i] = ctx[i];
    }

    Bytes sig(params.sig_bytes);
    std::size_t len = 0;
    pqc_fns(impl, params.id)
        ->signature_internal(sig.data(), &len, ptr(msg), msg.size(), pre, 2 + ctx.size(),
                             rnd.data(), sk.data());

    if (len != params.sig_bytes) {
        throw std::logic_error("pq-crystals produced a signature of unexpected length");
    }
    return sig;
}

bool verify(const ParamSet &params, ByteView pk, ByteView msg, ByteView ctx, ByteView sig,
            Impl impl) {
    require_known(params);
    require_available(impl);

    if (pk.size() != params.pk_bytes || sig.size() != params.sig_bytes ||
        ctx.size() > MAX_CONTEXT_BYTES) {
        return false;
    }

    if (impl == Impl::OpenSsl) {
        return detail::openssl_verify(params, pk, msg, ctx, sig);
    }

    return pqc_fns(impl, params.id)
               ->verify(sig.data(), sig.size(), ptr(msg), msg.size(), ptr(ctx), ctx.size(),
                        pk.data()) == 0;
}

bool verify_internal(const ParamSet &params, ByteView pk, ByteView mprime, ByteView sig,
                     Impl impl) {
    require_known(params);
    require_available(impl);

    if (pk.size() != params.pk_bytes || sig.size() != params.sig_bytes) {
        return false;
    }

    if (impl == Impl::OpenSsl) {
        return detail::openssl_verify_internal(params, pk, mprime, sig);
    }

    return pqc_fns(impl, params.id)
               ->verify_internal(sig.data(), sig.size(), ptr(mprime), mprime.size(), ptr({}), 0,
                                 pk.data()) == 0;
}

bool verify_mu(const ParamSet &params, ByteView pk, ByteView mu, ByteView sig, Impl impl) {
    require_known(params);
    if (!supports(impl, Interface::Mu)) {
        throw std::invalid_argument(std::string("CPU implementation '") + impl_name(impl) +
                                    "' cannot verify from mu");
    }
    require_size(mu, MU_BYTES, "mu");

    if (pk.size() != params.pk_bytes || sig.size() != params.sig_bytes) {
        return false;
    }

    if (impl == Impl::OpenSsl) {
        return detail::openssl_verify_mu(params, pk, mu, sig);
    }
    return sig_stages(params, pk, mu, sig).verdict == Verdict::Pass;
}

} // namespace ppq::cpuref
