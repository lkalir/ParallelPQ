#ifndef PPQ_CPUREF_OPENSSL_IMPL_H_
#define PPQ_CPUREF_OPENSSL_IMPL_H_

#include "ppq/cpuref.h"

namespace ppq::cpuref::detail {

/// False when built without OpenSSL; the other functions then throw std::logic_error
bool openssl_available();

KeyPair openssl_keygen(const ParamSet &params, ByteView xi);
Bytes openssl_sign(const ParamSet &params, ByteView sk, ByteView msg, ByteView ctx, ByteView rnd);
bool openssl_verify(const ParamSet &params, ByteView pk, ByteView msg, ByteView ctx, ByteView sig);
/// message-encoding 0: the message is M' itself
bool openssl_verify_internal(const ParamSet &params, ByteView pk, ByteView mprime, ByteView sig);
/// mu 1: the message is mu
bool openssl_verify_mu(const ParamSet &params, ByteView pk, ByteView mu, ByteView sig);

} // namespace ppq::cpuref::detail

#endif
