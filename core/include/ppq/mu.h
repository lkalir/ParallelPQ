#ifndef PPQ_MU_H_
#define PPQ_MU_H_

/// tr and mu (FIPS 204 Algorithms 7 and 8), for host and device

#include <cstddef>
#include <cstdint>

#include "ppq/hd.h"
#include "ppq/keccak.h"
#include "ppq/params.h"
#include "ppq/workload.h"

namespace ppq {

/// tr = H(pk, 64)
PPQ_HD inline void compute_tr(const std::uint8_t *pk, std::size_t pk_bytes, std::uint8_t *tr) {
    shake<Shake256>(pk, pk_bytes, tr, TR_BYTES);
}

/// mu for one signature through its interface:
///   external  H(tr || 0 || |ctx| || ctx || M, 64)
///   internal  H(tr || M', 64)
///   mu        the 64 message bytes themselves
PPQ_HD inline void compute_mu(Interface iface, const std::uint8_t *tr, const std::uint8_t *msg,
                              std::size_t msg_len, const std::uint8_t *ctx, std::size_t ctx_len,
                              std::uint8_t *mu) {
    if (iface == Interface::Mu) {
        for (std::size_t i = 0; i < MU_BYTES; ++i) {
            mu[i] = msg[i];
        }
        return;
    }

    Shake256 h;
    h.absorb(tr, TR_BYTES);
    if (iface == Interface::External) {
        const std::uint8_t prefix[2] = {0, static_cast<std::uint8_t>(ctx_len)};
        h.absorb(prefix, 2);
        h.absorb(ctx, ctx_len);
    }
    h.absorb(msg, msg_len);
    h.squeeze(mu, MU_BYTES);
}

} // namespace ppq

#endif
