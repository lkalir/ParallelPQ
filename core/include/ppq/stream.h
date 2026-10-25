#ifndef PPQ_STREAM_H_
#define PPQ_STREAM_H_

#include <utility>

#include "ppq/hip_check.h"

namespace ppq {

class Event {
  public:
    explicit Event(unsigned flags = hipEventDefault) {
        HIP_CHECK(hipEventCreateWithFlags(&_event, flags));
    }

    ~Event() {
        if (_event != nullptr) {
            (void)hipEventDestroy(_event);
        }
    }

    Event(const Event &) = delete;
    Event &operator=(const Event &) = delete;

    Event(Event &&other) noexcept : _event(std::exchange(other._event, nullptr)) {}

    Event &operator=(Event &&other) noexcept {
        if (this != &other) {
            if (_event != nullptr) {
                (void)hipEventDestroy(_event);
            }
            _event = std::exchange(other._event, nullptr);
        }
        return *this;
    }

    void record(hipStream_t stream) { HIP_CHECK(hipEventRecord(_event, stream)); }
    void synchronize() const { HIP_CHECK(hipEventSynchronize(_event)); }

    bool done() const {
        const hipError_t err = hipEventQuery(_event);
        if (err == hipErrorNotReady) {
            return false;
        }
        HIP_CHECK(err);
        return true;
    }

    hipEvent_t get() const { return _event; }

  private:
    hipEvent_t _event = nullptr;
};

class Stream {
  public:
    explicit Stream(unsigned flags = hipStreamNonBlocking) {
        HIP_CHECK(hipStreamCreateWithFlags(&_stream, flags));
    }

    ~Stream() {
        if (_stream != nullptr) {
            (void)hipStreamDestroy(_stream);
        }
    }

    Stream(const Stream &) = delete;
    Stream &operator=(const Stream &) = delete;

    Stream(Stream &&other) noexcept : _stream(std::exchange(other._stream, nullptr)) {}

    Stream &operator=(Stream &&other) noexcept {
        if (this != &other) {
            if (_stream != nullptr) {
                (void)hipStreamDestroy(_stream);
            }
            _stream = std::exchange(other._stream, nullptr);
        }
        return *this;
    }

    void synchronize() const { HIP_CHECK(hipStreamSynchronize(_stream)); }
    void wait(const Event &e) const { HIP_CHECK(hipStreamWaitEvent(_stream, e.get(), 0)); }

    hipStream_t get() const { return _stream; }

  private:
    hipStream_t _stream = nullptr;
};

/// Milliseconds from `from` to `to`; both must have completed
inline float elapsed_ms(const Event &from, const Event &to) {
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, from.get(), to.get()));
    return ms;
}

} // namespace ppq

#endif
