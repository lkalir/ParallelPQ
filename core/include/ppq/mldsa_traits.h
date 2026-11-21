#ifndef PPQ_MLDSA_TRAITS_H_
#define PPQ_MLDSA_TRAITS_H_

/// Compile-time parameter set constants for kernels and PPQ_HD code.
///
/// Device code may only use the integral members: they are evaluated here, in host context, so
/// no constexpr host function is ever called from device code (no --expt-relaxed-constexpr).

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "ppq/params.h"

namespace ppq {

template <int P> struct Mldsa {
  private:
    static constexpr ParamSet params = *find_param_set(P);

  public:
    static constexpr int id = P;
    static constexpr int k = params.k;
    static constexpr int l = params.l;
    static constexpr int tau = params.tau;
    static constexpr int omega = params.omega;
    static constexpr int z_bits = params.z_bits;
    static constexpr int w1_bits = params.w1_bits;
    static constexpr std::int32_t gamma1 = params.gamma1;
    static constexpr std::int32_t gamma2 = params.gamma2;
    static constexpr std::int32_t beta = params.beta;
    static constexpr std::size_t ctilde_bytes = params.ctilde_bytes;
    static constexpr std::size_t z_poly_bytes = params.z_poly_bytes;
    static constexpr std::size_t hint_bytes = params.hint_bytes;
    static constexpr std::size_t w1_poly_bytes = N * static_cast<std::size_t>(params.w1_bits) / 8;
    static constexpr std::size_t w1_bytes = params.w1_bytes;
    static constexpr std::size_t pk_bytes = params.pk_bytes;
    static constexpr std::size_t sig_bytes = params.sig_bytes;
};

/// Calls fn(Mldsa<id>{}) for a runtime parameter set id
template <class Fn> decltype(auto) with_params(int id, Fn &&fn) {
    switch (id) {
    case 44:
        return fn(Mldsa<44>{});
    case 65:
        return fn(Mldsa<65>{});
    case 87:
        return fn(Mldsa<87>{});
    default:
        throw std::invalid_argument("unknown ML-DSA parameter set " + std::to_string(id));
    }
}

} // namespace ppq

#endif
