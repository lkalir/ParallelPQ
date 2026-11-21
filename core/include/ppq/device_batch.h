#ifndef PPQ_DEVICE_BATCH_H_
#define PPQ_DEVICE_BATCH_H_

/// Device memory for the staged verifier: a chunk of a workload's signature inputs, its keys,
/// and the key cache and per-signature work buffers, each exposing the views in
/// verify_layout.h.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "ppq/buffer.h"
#include "ppq/verify_layout.h"
#include "ppq/workload.h"

namespace ppq {

/// Signatures [first, first + count) of a workload in device memory. Message and context
/// offsets are rebased to the chunk; key indices are copied as given (the workload's own, or a
/// remapping onto the chunk's key table).
class DeviceSigBatch {
  public:
    DeviceSigBatch(const Workload &w, std::size_t first, std::size_t count,
                   const std::vector<std::uint32_t> *key_index = nullptr)
        : _count(count), _iface(w.iface) {
        if (first + count > w.sig_count()) {
            throw std::out_of_range("signature chunk is outside the workload");
        }
        const auto f = static_cast<std::ptrdiff_t>(first);
        const auto e = static_cast<std::ptrdiff_t>(first + count);
        const std::size_t sb = w.params.sig_bytes;

        _sigs = DeviceBuffer<std::uint8_t>(
            std::vector<std::uint8_t>(w.sigs.begin() + f * static_cast<std::ptrdiff_t>(sb),
                                      w.sigs.begin() + e * static_cast<std::ptrdiff_t>(sb)));
        if (key_index != nullptr) {
            if (key_index->size() != count) {
                throw std::invalid_argument("key index remap must cover the chunk");
            }
            _key_index = DeviceBuffer<std::uint32_t>(*key_index);
        } else {
            _key_index = DeviceBuffer<std::uint32_t>(
                std::vector<std::uint32_t>(w.key_index.begin() + f, w.key_index.begin() + e));
        }
        upload_ragged(w.msg_offsets, w.msg_data, first, count, _msg_offsets, _msg_data);
        upload_ragged(w.ctx_offsets, w.ctx_data, first, count, _ctx_offsets, _ctx_data);
    }

    SigInputs view() const {
        return {_sigs.get(),
                _key_index.get(),
                _msg_offsets.get(),
                _msg_data.get(),
                _ctx_offsets.get(),
                _ctx_data.get(),
                _count,
                _iface};
    }

    std::size_t count() const { return _count; }

  private:
    static void upload_ragged(const std::vector<std::uint64_t> &offsets,
                              const std::vector<std::uint8_t> &data, std::size_t first,
                              std::size_t count, DeviceBuffer<std::uint64_t> &d_offsets,
                              DeviceBuffer<std::uint8_t> &d_data) {
        const std::uint64_t base = offsets[first];
        std::vector<std::uint64_t> rebased(count + 1);
        for (std::size_t i = 0; i <= count; ++i) {
            rebased[i] = offsets[first + i] - base;
        }
        d_offsets = DeviceBuffer<std::uint64_t>(rebased);
        // At least one byte so the pointer is never null, even for empty messages
        std::vector<std::uint8_t> bytes(data.begin() + static_cast<std::ptrdiff_t>(base),
                                        data.begin() +
                                            static_cast<std::ptrdiff_t>(offsets[first + count]));
        if (bytes.empty()) {
            bytes.assign(1, 0); // (not push_back: GCC 13 misreports it as -Wfree-nonheap-object)
        }
        d_data = DeviceBuffer<std::uint8_t>(bytes);
    }

    std::size_t _count;
    Interface _iface;
    DeviceBuffer<std::uint8_t> _sigs;
    DeviceBuffer<std::uint32_t> _key_index;
    DeviceBuffer<std::uint64_t> _msg_offsets;
    DeviceBuffer<std::uint8_t> _msg_data;
    DeviceBuffer<std::uint64_t> _ctx_offsets;
    DeviceBuffer<std::uint8_t> _ctx_data;
};

/// Public keys in device memory
class DeviceKeys {
  public:
    explicit DeviceKeys(const std::vector<std::uint8_t> &pks, std::size_t count)
        : _pks(pks.empty() ? std::vector<std::uint8_t>{0} : pks), _count(count) {}

    KeyInputs view() const { return {_pks.get(), _count}; }

  private:
    DeviceBuffer<std::uint8_t> _pks;
    std::size_t _count;
};

/// tr, A_hat and t1_hat for up to `capacity` keys
class KeyCacheBuffers {
  public:
    KeyCacheBuffers(const ParamSet &p, std::size_t capacity)
        : _strides(key_strides(p)), _capacity(capacity), _tr(capacity * _strides.tr),
          _a_hat(capacity * _strides.a_hat), _t1_hat(capacity * _strides.t1_hat) {}

    KeyCacheView view(std::size_t count) const {
        if (count > _capacity) {
            throw std::out_of_range("more keys than the key cache holds");
        }
        return {_tr.get(), _a_hat.get(), _t1_hat.get(), count};
    }

    std::size_t capacity() const { return _capacity; }
    const KeyStrides &strides() const { return _strides; }

  private:
    KeyStrides _strides;
    std::size_t _capacity;
    DeviceBuffer<std::uint8_t> _tr;
    DeviceBuffer<std::int32_t> _a_hat;
    DeviceBuffer<std::int32_t> _t1_hat;
};

/// Per-signature intermediates for up to `capacity` signatures
class SigWorkBuffers {
  public:
    SigWorkBuffers(const ParamSet &p, std::size_t capacity)
        : _strides(work_strides(p)), _capacity(capacity), _status(capacity),
          _z(capacity * _strides.z), _h(capacity * _strides.h), _c(capacity * _strides.c),
          _mu(capacity * _strides.mu), _w(capacity * _strides.w), _w1(capacity * _strides.w1) {}

    SigWork view() const {
        return {_status.get(), _z.get(), _h.get(), _c.get(), _mu.get(), _w.get(), _w1.get()};
    }

    std::size_t capacity() const { return _capacity; }
    const WorkStrides &strides() const { return _strides; }

    // Downloads for tests and debugging (count signatures)
    std::vector<std::uint32_t> status(std::size_t count) const { return get(_status, count); }
    std::vector<std::int32_t> z(std::size_t count) const { return get(_z, count * _strides.z); }
    std::vector<std::uint8_t> h(std::size_t count) const { return get(_h, count * _strides.h); }
    std::vector<std::int32_t> c(std::size_t count) const { return get(_c, count * _strides.c); }
    std::vector<std::uint8_t> mu(std::size_t count) const { return get(_mu, count * _strides.mu); }
    std::vector<std::int32_t> w(std::size_t count) const { return get(_w, count * _strides.w); }
    std::vector<std::uint8_t> w1(std::size_t count) const { return get(_w1, count * _strides.w1); }

  private:
    template <class T> static std::vector<T> get(const DeviceBuffer<T> &b, std::size_t n) {
        std::vector<T> out(n);
        if (n > 0) {
            HIP_CHECK(hipMemcpy(out.data(), b.get(), n * sizeof(T), hipMemcpyDeviceToHost));
        }
        return out;
    }

    WorkStrides _strides;
    std::size_t _capacity;
    DeviceBuffer<std::uint32_t> _status;
    DeviceBuffer<std::int32_t> _z;
    DeviceBuffer<std::uint8_t> _h;
    DeviceBuffer<std::int32_t> _c;
    DeviceBuffer<std::uint8_t> _mu;
    DeviceBuffer<std::int32_t> _w;
    DeviceBuffer<std::uint8_t> _w1;
};

} // namespace ppq

#endif
