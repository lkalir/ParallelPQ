#ifndef PPQ_CPUREF_RANDOM_SOURCE_H_
#define PPQ_CPUREF_RANDOM_SOURCE_H_

#include <cstddef>
#include <cstdint>
#include <span>

namespace ppq::cpuref::detail {

struct RandomSource {
    const std::uint8_t *next = nullptr;
    std::size_t left = 0;
};

/// While alive, randombytes() on this thread hands out exactly these bytes, in order.
/// The pq-crystals code draws its keygen seed through randombytes(); this makes key
/// generation a pure function of the seed. Asking for more bytes than supplied aborts.
class ScopedRandomBytes {
  public:
    explicit ScopedRandomBytes(std::span<const std::uint8_t> bytes);
    ~ScopedRandomBytes();

    ScopedRandomBytes(const ScopedRandomBytes &) = delete;
    ScopedRandomBytes &operator=(const ScopedRandomBytes &) = delete;

    std::size_t left() const { return _source.left; }

  private:
    RandomSource _source;
    RandomSource *_previous;
};

} // namespace ppq::cpuref::detail

#endif
