#ifndef PPQ_HASH_BATCH_H_
#define PPQ_HASH_BATCH_H_

#include <cstddef>
#include <cstdint>

#include "ppq/buffer.h"
#include "ppq/host_buffer.h"
#include "ppq/mldsa.h"

namespace ppq {

/// Inputs and outputs for `count` signatures in pinned host memory
class HostBatch {
  public:
    HostBatch(const BatchLayout &layout, std::size_t count)
        : _layout(layout), _count(count), _pk(count * layout.pk_stride()),
          _msg(count * layout.msg_stride()), _digest(count * layout.digest_stride()),
          _matrix(count * layout.matrix_stride()) {}

    void fill_random(std::uint32_t seed) {
        fill_random_inputs(_layout, _pk.get(), _msg.get(), _count, seed);
    }

    BatchView view() const { return {_pk.get(), _msg.get(), _digest.get(), _matrix.get(), _count}; }

    std::uint8_t *pk() { return _pk.get(); }
    std::uint8_t *msg() { return _msg.get(); }

    const BatchLayout &layout() const { return _layout; }
    std::size_t count() const { return _count; }

  private:
    BatchLayout _layout;
    std::size_t _count;
    PinnedBuffer<std::uint8_t> _pk;
    PinnedBuffer<std::uint8_t> _msg;
    PinnedBuffer<std::uint8_t> _digest;
    PinnedBuffer<std::uint8_t> _matrix;
};

/// Device buffers for up to `capacity` signatures
class DeviceBatch {
  public:
    DeviceBatch(const BatchLayout &layout, std::size_t capacity)
        : _layout(layout), _capacity(capacity), _pk(capacity * layout.pk_stride()),
          _msg(capacity * layout.msg_stride()), _digest(capacity * layout.digest_stride()),
          _matrix(capacity * layout.matrix_stride()) {}

    BatchView view(std::size_t count) const {
        return {_pk.get(), _msg.get(), _digest.get(), _matrix.get(), count};
    }

    /// Queues the input copy for `count` signatures starting at host signature `first`
    void upload_async(const HostBatch &host, std::size_t first, std::size_t count,
                      hipStream_t stream) {
        const BatchView h = slice(_layout, host.view(), first, count);
        HIP_CHECK(hipMemcpyAsync(_pk.get(), h.pk, count * _layout.pk_stride(),
                                 hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(_msg.get(), h.msg, count * _layout.msg_stride(),
                                 hipMemcpyHostToDevice, stream));
    }

    void download_async(const HostBatch &host, std::size_t first, std::size_t count,
                        hipStream_t stream) const {
        const BatchView h = slice(_layout, host.view(), first, count);
        HIP_CHECK(hipMemcpyAsync(h.digest, _digest.get(), count * _layout.digest_stride(),
                                 hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipMemcpyAsync(h.matrix, _matrix.get(), count * _layout.matrix_stride(),
                                 hipMemcpyDeviceToHost, stream));
    }

    std::size_t capacity() const { return _capacity; }

  private:
    BatchLayout _layout;
    std::size_t _capacity;
    DeviceBuffer<std::uint8_t> _pk;
    DeviceBuffer<std::uint8_t> _msg;
    DeviceBuffer<std::uint8_t> _digest;
    DeviceBuffer<std::uint8_t> _matrix;
};

} // namespace ppq

#endif
