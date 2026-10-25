#ifndef PPQ_HIP_CHECK_H_
#define PPQ_HIP_CHECK_H_

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

namespace ppq {

class HipError : public std::runtime_error {
  public:
    HipError(hipError_t code, const std::string &what)
        : std::runtime_error(what + " failed: " + hipGetErrorName(code) + " (" +
                             hipGetErrorString(code) + ")"),
          _code(code) {}

    hipError_t code() const { return _code; }

  private:
    hipError_t _code;
};

inline void check_hip(hipError_t err, const char *what) {
    if (err != hipSuccess) {
        throw HipError(err, what);
    }
}

} // namespace ppq

/// Stringifies the call so the message says exactly what failed.
#define HIP_CHECK(call) ::ppq::check_hip((call), #call)

#endif
