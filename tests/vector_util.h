#ifndef TESTS_VECTOR_UTIL_H_
#define TESTS_VECTOR_UTIL_H_

/// Helpers for tests that read tests/vectors/{acvp,stages}

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "ppq/kv_vectors.h"
#include "ppq/params.h"
#include "ppq/workload.h"

namespace vector_util {

using Poly = std::array<std::int32_t, ppq::N>;

inline const ppq::ParamSet &params_of(const ppq::KvRecord &r) {
    const ppq::ParamSet *p = ppq::find_param_set(static_cast<int>(r.integer("params")));
    if (p == nullptr) {
        throw std::runtime_error(r.where() + ": unknown parameter set");
    }
    return *p;
}

/// A field of `count` concatenated polynomials
inline std::vector<Poly> polys(const ppq::KvRecord &r, const std::string &key, std::size_t count) {
    const std::vector<std::int32_t> flat = r.ints(key);
    if (flat.size() != count * ppq::N) {
        throw std::runtime_error(r.where() + ": '" + key + "' should hold " +
                                 std::to_string(count) + " polynomials");
    }
    std::vector<Poly> out(count);
    for (std::size_t i = 0; i < count; ++i) {
        std::copy(flat.begin() + static_cast<std::ptrdiff_t>(i * ppq::N),
                  flat.begin() + static_cast<std::ptrdiff_t>((i + 1) * ppq::N), out[i].begin());
    }
    return out;
}

inline Poly poly(const ppq::KvRecord &r, const std::string &key) { return polys(r, key, 1)[0]; }

inline ppq::Verdict parse_verdict(const ppq::KvRecord &r) {
    const std::string &s = r.str("verdict");
    for (std::size_t i = 0; i < ppq::VERDICT_COUNT; ++i) {
        const auto v = static_cast<ppq::Verdict>(i);
        if (s == ppq::verdict_name(v)) {
            return v;
        }
    }
    throw std::runtime_error(r.where() + ": unknown verdict '" + s + "'");
}

} // namespace vector_util

#endif
