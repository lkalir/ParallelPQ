#ifndef TESTS_MLDSA_VECTORS_H_
#define TESTS_MLDSA_VECTORS_H_

#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ppq/shake_vectors.h"

namespace mldsa_test {

using Bytes = std::vector<std::uint8_t>;

/// tests/vectors/mldsa_hash.txt from tools/kat.py (hashlib): tr = H(pk) and
/// mu = H(tr || 0 || 0 || M). The ExpandA stream digests in the file are from module 6's
/// fixed 5-block stream and are no longer checked.
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

} // namespace mldsa_test

#endif
