#ifndef PPQ_HOST_BUFFER_H_
#define PPQ_HOST_BUFFER_H_

#include <cstddef>
#include <utility>

#include "ppq/hip_check.h"

namespace ppq {

/// Page-locked host memory, required for hipMemcpyAsync to overlap with kernels
template <class T> class PinnedBuffer {
  public:
    PinnedBuffer() = default;

    explicit PinnedBuffer(std::size_t count) : _count(count) {
        if (_count > 0) {
            HIP_CHECK(
                hipHostMalloc(reinterpret_cast<void **>(&_ptr), bytes(), hipHostMallocDefault));
        }
    }

    ~PinnedBuffer() { release(); }

    PinnedBuffer(const PinnedBuffer &) = delete;
    PinnedBuffer &operator=(const PinnedBuffer &) = delete;

    PinnedBuffer(PinnedBuffer &&other) noexcept
        : _ptr(std::exchange(other._ptr, nullptr)), _count(std::exchange(other._count, 0)) {}

    PinnedBuffer &operator=(PinnedBuffer &&other) noexcept {
        if (this != &other) {
            release();
            _ptr = std::exchange(other._ptr, nullptr);
            _count = std::exchange(other._count, 0);
        }
        return *this;
    }

    T *get() const { return _ptr; }
    std::size_t size() const { return _count; }
    std::size_t bytes() const { return _count * sizeof(T); }

  private:
    void release() noexcept {
        if (_ptr != nullptr) {
            (void)hipHostFree(_ptr);
            _ptr = nullptr;
        }
    }

    T *_ptr = nullptr;
    std::size_t _count = 0;
};

} // namespace ppq

#endif
