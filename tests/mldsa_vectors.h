#ifndef TESTS_MLDSA_VECTORS_H_
#define TESTS_MLDSA_VECTORS_H_

#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ppq/mldsa.h"
#include "ppq/shake_vectors.h"

namespace mldsa_test {

using Bytes = std::vector<std::uint8_t>;

constexpr std::size_t STREAM_DIGEST_BYTES = 32;

struct Vector {
    Bytes pk;
    Bytes msg;
    Bytes tr;
    Bytes mu;
    Bytes stream_digests;
};

/// Vectors grouped by parameter set id; every vector in a group has the same message length
inline std::map<int, std::vector<Vector>> load(const std::string &path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot open " + path);
    }

    std::map<int, std::vector<Vector>> groups;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream is(line);
        int id = 0;
        std::string pk, msg, tr, mu, digests;

        if (!(is >> id >> pk >> msg >> tr >> mu >> digests)) {
            throw std::runtime_error("malformed line in " + path);
        }

        groups[id].push_back({ppq::from_hex(pk), ppq::from_hex(msg), ppq::from_hex(tr),
                              ppq::from_hex(mu), ppq::from_hex(digests)});
    }

    return groups;
}

/// SHAKE256-32 of each EXPAND_A_BYTES stream, matching the vector file
inline Bytes stream_digests(const std::uint8_t *matrix, std::size_t entries) {
    Bytes out;

    for (std::size_t e = 0; e < entries; ++e) {
        const Bytes d =
            ppq::shake_cpu(ppq::ShakeVariant::Shake256, matrix + e * ppq::EXPAND_A_BYTES,
                           ppq::EXPAND_A_BYTES, STREAM_DIGEST_BYTES);
        out.insert(out.end(), d.begin(), d.end());
    }

    return out;
}

/// Copies one vector's key and M' = 0 || 0 || M into packed batch arrays at index s
inline void pack(const ppq::BatchLayout &layout, const Vector &v, std::size_t s, std::uint8_t *pk,
                 std::uint8_t *msg) {
    std::copy(v.pk.begin(), v.pk.end(), pk + s * layout.pk_stride());
    std::uint8_t *m = msg + s * layout.msg_stride();
    m[0] = 0;
    m[1] = 0;
    std::copy(v.msg.begin(), v.msg.end(), m + ppq::MESSAGE_PREFIX_BYTES);
}

inline Bytes bytes(const std::uint8_t *p, std::size_t n) { return Bytes(p, p + n); }

} // namespace mldsa_test

#endif
