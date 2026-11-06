/// OpenSSL 3.5+ ML-DSA backend (EVP_SIGNATURE-ML-DSA).

#include <stdexcept>

#include "openssl_impl.h"

#if PPQ_CPUREF_HAVE_OPENSSL

#include <memory>
#include <string>

#include <openssl/core_names.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/params.h>

namespace ppq::cpuref::detail {
namespace {

struct PkeyFree {
    void operator()(EVP_PKEY *p) const { EVP_PKEY_free(p); }
};
struct CtxFree {
    void operator()(EVP_PKEY_CTX *p) const { EVP_PKEY_CTX_free(p); }
};
struct SigFree {
    void operator()(EVP_SIGNATURE *p) const { EVP_SIGNATURE_free(p); }
};
using Pkey = std::unique_ptr<EVP_PKEY, PkeyFree>;
using Ctx = std::unique_ptr<EVP_PKEY_CTX, CtxFree>;
using Sig = std::unique_ptr<EVP_SIGNATURE, SigFree>;

[[noreturn]] void fail(const std::string &what) {
    const unsigned long code = ERR_get_error();
    char buf[256] = "no OpenSSL error queued";
    if (code != 0) {
        ERR_error_string_n(code, buf, sizeof(buf));
    }
    ERR_clear_error();
    throw std::runtime_error("OpenSSL " + what + " failed: " + buf);
}

/// Fetched once per parameter set; EVP_SIGNATURE objects are safe to share across threads
EVP_SIGNATURE *algorithm(const ParamSet &params) {
    static const Sig ALGS[] = {Sig(EVP_SIGNATURE_fetch(nullptr, "ML-DSA-44", nullptr)),
                               Sig(EVP_SIGNATURE_fetch(nullptr, "ML-DSA-65", nullptr)),
                               Sig(EVP_SIGNATURE_fetch(nullptr, "ML-DSA-87", nullptr))};
    const int index = params.id == 44 ? 0 : params.id == 65 ? 1 : 2;
    return ALGS[index].get();
}

/// OSSL_PARAM wants non-const pointers even for inputs it only reads
void *mut(ByteView v) {
    static std::uint8_t empty = 0;
    return v.empty() ? &empty : const_cast<std::uint8_t *>(v.data());
}

} // namespace

bool openssl_available() {
    static const bool ok = [] {
        for (const ParamSet &p : PARAM_SETS) {
            if (algorithm(p) == nullptr) {
                ERR_clear_error();
                return false;
            }
        }
        return true;
    }();
    return ok;
}

KeyPair openssl_keygen(const ParamSet &params, ByteView xi) {
    Ctx ctx(EVP_PKEY_CTX_new_from_name(nullptr, params.name, nullptr));
    if (!ctx || EVP_PKEY_keygen_init(ctx.get()) != 1) {
        fail("keygen init");
    }

    const OSSL_PARAM settings[] = {
        OSSL_PARAM_construct_octet_string(OSSL_PKEY_PARAM_ML_DSA_SEED, mut(xi), xi.size()),
        OSSL_PARAM_construct_end(),
    };
    if (EVP_PKEY_CTX_set_params(ctx.get(), settings) != 1) {
        fail("keygen seed");
    }

    EVP_PKEY *raw = nullptr;
    if (EVP_PKEY_generate(ctx.get(), &raw) != 1) {
        fail("keygen");
    }
    const Pkey key(raw);

    KeyPair kp{Bytes(params.pk_bytes), Bytes(params.sk_bytes)};
    std::size_t pk_len = kp.pk.size();
    std::size_t sk_len = kp.sk.size();
    if (EVP_PKEY_get_raw_public_key(key.get(), kp.pk.data(), &pk_len) != 1 ||
        EVP_PKEY_get_raw_private_key(key.get(), kp.sk.data(), &sk_len) != 1 ||
        pk_len != params.pk_bytes || sk_len != params.sk_bytes) {
        fail("key export");
    }
    return kp;
}

Bytes openssl_sign(const ParamSet &params, ByteView sk, ByteView msg, ByteView ctx, ByteView rnd) {
    const Pkey key(
        EVP_PKEY_new_raw_private_key_ex(nullptr, params.name, nullptr, sk.data(), sk.size()));
    if (!key) {
        fail("private key import");
    }

    const Ctx pctx(EVP_PKEY_CTX_new_from_pkey(nullptr, key.get(), nullptr));
    const OSSL_PARAM settings[] = {
        OSSL_PARAM_construct_octet_string(OSSL_SIGNATURE_PARAM_CONTEXT_STRING, mut(ctx),
                                          ctx.size()),
        OSSL_PARAM_construct_octet_string(OSSL_SIGNATURE_PARAM_TEST_ENTROPY, mut(rnd), rnd.size()),
        OSSL_PARAM_construct_end(),
    };
    if (!pctx || EVP_PKEY_sign_message_init(pctx.get(), algorithm(params), settings) != 1) {
        fail("sign init");
    }

    Bytes sig(params.sig_bytes);
    std::size_t len = sig.size();
    if (EVP_PKEY_sign(pctx.get(), sig.data(), &len, static_cast<const unsigned char *>(mut(msg)),
                      msg.size()) != 1 ||
        len != params.sig_bytes) {
        fail("sign");
    }
    return sig;
}

namespace {

/// Verifies `msg` (M, M' or mu, depending on settings) against a raw public key
bool verify_with(const ParamSet &params, ByteView pk, ByteView msg, ByteView sig,
                 const OSSL_PARAM *settings) {
    const Pkey key(
        EVP_PKEY_new_raw_public_key_ex(nullptr, params.name, nullptr, pk.data(), pk.size()));
    if (!key) {
        ERR_clear_error();
        return false;
    }

    const Ctx pctx(EVP_PKEY_CTX_new_from_pkey(nullptr, key.get(), nullptr));
    if (!pctx || EVP_PKEY_verify_message_init(pctx.get(), algorithm(params), settings) != 1) {
        fail("verify init");
    }

    const int rc = EVP_PKEY_verify(pctx.get(), sig.data(), sig.size(),
                                   static_cast<const unsigned char *>(mut(msg)), msg.size());
    ERR_clear_error();
    return rc == 1;
}

} // namespace

bool openssl_verify(const ParamSet &params, ByteView pk, ByteView msg, ByteView ctx, ByteView sig) {
    const OSSL_PARAM settings[] = {
        OSSL_PARAM_construct_octet_string(OSSL_SIGNATURE_PARAM_CONTEXT_STRING, mut(ctx),
                                          ctx.size()),
        OSSL_PARAM_construct_end(),
    };
    return verify_with(params, pk, msg, sig, settings);
}

bool openssl_verify_internal(const ParamSet &params, ByteView pk, ByteView mprime, ByteView sig) {
    int raw = 0;
    const OSSL_PARAM settings[] = {
        OSSL_PARAM_construct_int(OSSL_SIGNATURE_PARAM_MESSAGE_ENCODING, &raw),
        OSSL_PARAM_construct_end(),
    };
    return verify_with(params, pk, mprime, sig, settings);
}

bool openssl_verify_mu(const ParamSet &params, ByteView pk, ByteView mu, ByteView sig) {
    int is_mu = 1;
    const OSSL_PARAM settings[] = {
        OSSL_PARAM_construct_int(OSSL_SIGNATURE_PARAM_MU, &is_mu),
        OSSL_PARAM_construct_end(),
    };
    return verify_with(params, pk, mu, sig, settings);
}

} // namespace ppq::cpuref::detail

#else

namespace ppq::cpuref::detail {

bool openssl_available() { return false; }

KeyPair openssl_keygen(const ParamSet &, ByteView) {
    throw std::logic_error("built without OpenSSL");
}

Bytes openssl_sign(const ParamSet &, ByteView, ByteView, ByteView, ByteView) {
    throw std::logic_error("built without OpenSSL");
}

bool openssl_verify(const ParamSet &, ByteView, ByteView, ByteView, ByteView) {
    throw std::logic_error("built without OpenSSL");
}

bool openssl_verify_internal(const ParamSet &, ByteView, ByteView, ByteView) {
    throw std::logic_error("built without OpenSSL");
}

bool openssl_verify_mu(const ParamSet &, ByteView, ByteView, ByteView) {
    throw std::logic_error("built without OpenSSL");
}

} // namespace ppq::cpuref::detail

#endif
