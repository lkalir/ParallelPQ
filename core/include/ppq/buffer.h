#ifndef PPQ_BUFFER_H_
#define PPQ_BUFFER_H_

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "ppq/hip_check.h"

namespace ppq {

template <class T> class DeviceBuffer {
  public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) : _count(count) {
        if (_count > 0) {
            HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&_ptr), bytes()));
        }
    }

    explicit DeviceBuffer(const std::vector<T> &host) : DeviceBuffer(host.size()) { upload(host); }

    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    DeviceBuffer(DeviceBuffer &&other) noexcept
        : _ptr(std::exchange(other._ptr, nullptr)), _count(std::exchange(other._count, 0)) {}

    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
        if (this != &other) {
            release();
            _ptr = std::exchange(other._ptr, nullptr);
            _count = std::exchange(other._count, 0);
        }
        return *this;
    }

    void upload(const std::vector<T> &host) {
        HIP_CHECK(hipMemcpy(_ptr, host.data(), std::min(host.size(), _count) * sizeof(T),
                            hipMemcpyHostToDevice));
    }

    /// Copies the first host.size() elements back.
    void download(std::vector<T> &host) const {
        HIP_CHECK(hipMemcpy(host.data(), _ptr, std::min(host.size(), _count) * sizeof(T),
                            hipMemcpyDeviceToHost));
    }

    T *get() const { return _ptr; }
    std::size_t size() const { return _count; }
    std::size_t bytes() const { return _count * sizeof(T); }

  private:
    void release() noexcept {
        if (_ptr != nullptr) {
            // never throw from a dtor
            (void)hipFree(_ptr);
            _ptr = nullptr;
        }
    }

    T *_ptr = nullptr;
    std::size_t _count = 0;
};

} // namespace ppq

#endif
