#ifndef PPQ_WORKLOAD_H_
#define PPQ_WORKLOAD_H_

/// A verification workload: unique public keys, and signatures that each point at a key.
///
/// Everything is stored in flat arrays so a batch can be copied to the GPU as-is. Messages and
/// contexts vary in length and are stored back to back with offsets: entry i of a ragged array
/// is data[offsets[i], offsets[i + 1]).
///
/// A workload is verified through one FIPS 204 entry point (its Interface), which decides what
/// the message field holds.
///
/// Workloads are saved as .ppq files, laid out as (all integers little-endian):
///
///   "PPQ1" | u32 param id | u32 interface | u64 keys | u64 signatures | u64 message bytes |
///   u64 context bytes | u64 seed |
///   public keys | signatures | u32 key index per signature |
///   u64 message offsets (signatures + 1) | message data |
///   u64 context offsets (signatures + 1) | context data |
///   u8 expected verdict per signature | u8 tamper per signature |
///   SHAKE256-256 checksum of everything before it

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ppq/params.h"

namespace ppq {

using ByteView = std::span<const std::uint8_t>;

/// Which verification entry point a workload's signatures are checked through. Values are
/// stored in .ppq files; append only.
enum class Interface : std::uint8_t {
    /// ML-DSA.Verify(pk, M, sigma, ctx): pure ML-DSA; messages are M, with contexts
    External = 0,
    /// ML-DSA.Verify_internal(pk, M', sigma): messages are M' itself; no contexts
    Internal = 1,
    /// Verify_internal given mu = H(tr || M', 64) (ACVP externalMu): messages are 64-byte mu
    Mu = 2,
};
constexpr std::size_t INTERFACE_COUNT = 3;

/// Why a signature fails (or not). Values are stored in .ppq files; append only.
enum class Verdict : std::uint8_t {
    Pass = 0,
    /// c-tilde does not match H(mu || w1Encode(w1')): wrong message, context, key or commitment
    BadCommitment = 1,
    /// ||z||_inf >= gamma1 - beta
    ZNorm = 2,
    /// The hint encoding is malformed (HintBitUnpack returns bottom)
    BadHint = 3,
};
constexpr std::size_t VERDICT_COUNT = 4;

/// How the generator corrupted a signature. Values are stored in .ppq files; append only.
enum class Tamper : std::uint8_t {
    None = 0,
    /// One message byte flipped after signing (or a byte added to an empty message)
    Message = 1,
    /// One bit of c-tilde flipped
    Commitment = 2,
    /// One z coefficient set to gamma1, outside the norm bound
    ZNorm = 3,
    /// The hint section made malformed
    Hint = 4,
};
constexpr std::size_t TAMPER_COUNT = 5;

const char *interface_name(Interface iface);
std::optional<Interface> parse_interface(std::string_view name);
const char *verdict_name(Verdict v);
std::optional<Verdict> parse_verdict(std::string_view name);
const char *tamper_name(Tamper t);
std::optional<Tamper> parse_tamper(std::string_view name);

/// The verdict a correct verifier must report for a signature tampered this way
Verdict expected_verdict(Tamper t);

struct Workload {
    ParamSet params = ML_DSA_44;
    Interface iface = Interface::External;
    /// Seed the workload was generated from; 0 when imported from elsewhere
    std::uint64_t seed = 0;

    std::vector<std::uint8_t> pks;
    std::vector<std::uint8_t> sigs;
    std::vector<std::uint32_t> key_index;
    std::vector<std::uint64_t> msg_offsets{0};
    std::vector<std::uint8_t> msg_data;
    std::vector<std::uint64_t> ctx_offsets{0};
    std::vector<std::uint8_t> ctx_data;
    std::vector<Verdict> expected;
    std::vector<Tamper> tamper;

    std::size_t key_count() const { return pks.size() / params.pk_bytes; }
    std::size_t sig_count() const { return key_index.size(); }
    std::size_t pass_count() const;

    ByteView pk(std::size_t key) const;
    ByteView sig(std::size_t i) const;
    ByteView msg(std::size_t i) const;
    ByteView ctx(std::size_t i) const;
    /// The public key signature i is checked against
    ByteView pk_of(std::size_t i) const { return pk(key_index[i]); }

    /// Returns the new key's index
    std::uint32_t add_key(ByteView pk);
    void add_signature(std::uint32_t key, ByteView sig, ByteView msg, ByteView ctx,
                       Verdict expected, Tamper tamper = Tamper::None);

    /// Throws std::runtime_error naming the first inconsistency
    void validate() const;
};

bool same_workload(const Workload &a, const Workload &b);

/// Writes to `path` (via a temporary file renamed into place). Throws std::runtime_error.
void save_workload(const Workload &w, const std::string &path);

/// Reads and validates a .ppq file, including its checksum. Throws std::runtime_error.
Workload load_workload(const std::string &path);

} // namespace ppq

#endif
