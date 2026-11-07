#ifndef PPQ_CPUREF_H_
#define PPQ_CPUREF_H_

/// CPU ML-DSA implementations from third-party libraries.
///
/// These are the reference ParallelPQ is measured and checked against. Key generation and
/// signing exist only to produce test workloads; ParallelPQ's own code verifies.
///
/// - Ref:     pq-crystals reference C code (portable)
/// - Avx2:    pq-crystals AVX2 code (x86-64 with AVX2, when built with PPQ_CPUREF_AVX2)
/// - OpenSsl: OpenSSL 3.5+ (when built with PPQ_WITH_OPENSSL and found)

#include <cstdint>
#include <span>
#include <vector>

#include "ppq/params.h"
#include "ppq/workload.h"

namespace ppq::cpuref {

enum class Impl : std::uint8_t {
    Ref,
    Avx2,
    OpenSsl,
};

constexpr Impl ALL_IMPLS[] = {Impl::Ref, Impl::Avx2, Impl::OpenSsl};

/// Verification entry points (defined with the workload model)
using ppq::Interface;
using ppq::interface_name;

const char *impl_name(Impl impl);

/// Compiled in and usable on this CPU
bool available(Impl impl);

std::vector<Impl> available_impls();

/// Available and able to verify through this interface (pq-crystals AVX2 has no mu entry)
bool supports(Impl impl, Interface iface);

using Bytes = std::vector<std::uint8_t>;
using ByteView = std::span<const std::uint8_t>;

struct KeyPair {
    Bytes pk;
    Bytes sk;
};

/// ML-DSA.KeyGen_internal(xi): the same seed always gives the same key pair.
/// Throws std::invalid_argument if xi is not SEED_BYTES long or the impl is unavailable.
KeyPair keygen(const ParamSet &params, ByteView xi, Impl impl = Impl::Ref);

/// ML-DSA.Sign (pure, with context) using the given 32 bytes as the signing randomness;
/// all-zero rnd gives the deterministic variant. Throws on bad sizes or ctx over 255 bytes.
Bytes sign(const ParamSet &params, ByteView sk, ByteView msg, ByteView ctx, ByteView rnd,
           Impl impl = Impl::Ref);

/// ML-DSA.Verify (pure, with context). Malformed input is a failed verification, not an error.
bool verify(const ParamSet &params, ByteView pk, ByteView msg, ByteView ctx, ByteView sig,
            Impl impl = Impl::Ref);

/// ML-DSA.Verify_internal: mprime is M' as signed, with no prefix added
bool verify_internal(const ParamSet &params, ByteView pk, ByteView mprime, ByteView sig,
                     Impl impl = Impl::Ref);

/// Verification given mu = H(tr || M', 64). Throws std::invalid_argument if the impl does not
/// support it, or mu is not 64 bytes.
bool verify_mu(const ParamSet &params, ByteView pk, ByteView mu, ByteView sig,
               Impl impl = Impl::Ref);

} // namespace ppq::cpuref

#endif
