/// GPU builds only: lists the devices through the ppq_core runtime layer. Compiled in the GPU
/// language, like module 6's app sources, so ppq.exe links the same way as before.

#include <exception>

#include "gpu_query.h"

namespace ppq::app {

GpuQuery query_gpus() {
    GpuQuery q;
    q.platform = platform_name();
    try {
        const int count = device_count();
        for (int id = 0; id < count; ++id) {
            q.devices.push_back(open_device(id));
        }
        if (count > 0) {
            open_device(0);
        }
    } catch (const std::exception &e) {
        q.error = e.what();
    }
    return q;
}

} // namespace ppq::app
