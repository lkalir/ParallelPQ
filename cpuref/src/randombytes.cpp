/// ParallelPQ's randombytes() for the pq-crystals code, replacing its randombytes.c.
///
/// There is deliberately no fallback to system randomness: an unexpected call means the
/// library changed how it draws randomness, and silently using fresh entropy would make
/// workloads irreproducible without anyone noticing.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "random_source.h"

namespace ppq::cpuref::detail {
namespace {

thread_local RandomSource *t_source = nullptr;

} // namespace

ScopedRandomBytes::ScopedRandomBytes(std::span<const std::uint8_t> bytes)
    : _source{bytes.data(), bytes.size()}, _previous(t_source) {
    t_source = &_source;
}

ScopedRandomBytes::~ScopedRandomBytes() { t_source = _previous; }

} // namespace ppq::cpuref::detail

extern "C" void randombytes(std::uint8_t *out, std::size_t outlen) {
    ppq::cpuref::detail::RandomSource *s = ppq::cpuref::detail::t_source;

    if (s == nullptr || outlen > s->left) {
        std::fprintf(stderr,
                     "ppq: randombytes(%zu) called with %zu seeded bytes available; "
                     "the CPU reference must only draw randomness ParallelPQ supplies\n",
                     outlen, s == nullptr ? std::size_t{0} : s->left);
        std::abort();
    }

    if (outlen > 0) {
        std::memcpy(out, s->next, outlen);
    }
    s->next += outlen;
    s->left -= outlen;
}
