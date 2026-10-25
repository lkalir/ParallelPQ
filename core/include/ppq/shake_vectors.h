#ifndef PPQ_SHAKE_VECTORS_H_
#define PPQ_SHAKE_VECTORS_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ppq {

enum class ShakeVariant : std::uint8_t {
    Shake128,
    Shake256,
};

const char *variant_name(ShakeVariant v);
unsigned variant_rate(ShakeVariant v);

struct ShakeVector {
    ShakeVariant variant = ShakeVariant::Shake128;
    std::vector<std::uint8_t> input;
    std::size_t out_len = 0;
    std::vector<std::uint8_t> expected;
};

/// Lines of "shake128|shake256 <input hex or -> <out_len> <output hex>"; '#' starts a comment.
/// Throws std::runtime_error on a malformed file.
std::vector<ShakeVector> load_shake_vectors(const std::string &path);

struct CavpFile {
    std::string path;
    ShakeVariant variant;
};

/// SHAKE{128,256}{ShortMsg,LongMsg,VariableOut}.rsp files present in `dir`
std::vector<CavpFile> find_cavp_files(const std::string &dir);

/// Byte-oriented entries of a NIST CAVP SHAKE .rsp file (Len and Outputlen are in bits)
std::vector<ShakeVector> load_cavp_vectors(const CavpFile &file);

std::vector<std::uint8_t> shake_cpu(ShakeVariant v, const std::uint8_t *in, std::size_t in_len,
                                    std::size_t out_len);

std::vector<std::uint8_t> from_hex(std::string_view hex);
std::string to_hex(const std::uint8_t *data, std::size_t len);

} // namespace ppq

#endif
