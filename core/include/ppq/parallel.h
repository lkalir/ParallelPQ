#ifndef PPQ_PARALLEL_H_
#define PPQ_PARALLEL_H_

/// A minimal host thread pool for embarrassingly parallel loops (workload generation and
/// the CPU verification baseline).

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace ppq {

/// 0 means "all hardware threads"
inline unsigned resolve_threads(unsigned requested) {
    if (requested != 0) {
        return requested;
    }
    const unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1 : hw;
}

/// Calls fn(i) for every i in [0, n) on up to `threads` threads, handing out indices in
/// chunks of `grain`. Rethrows the first exception after all threads stop.
template <class Fn>
void parallel_for(std::size_t n, unsigned threads, Fn &&fn, std::size_t grain = 16) {
    if (n == 0) {
        return;
    }

    grain = std::max<std::size_t>(grain, 1);
    const std::size_t chunks = (n + grain - 1) / grain;
    const auto workers =
        static_cast<unsigned>(std::min<std::size_t>(resolve_threads(threads), chunks));

    std::atomic<std::size_t> next{0};
    std::atomic<bool> stop{false};
    std::exception_ptr error;
    std::mutex error_mutex;

    const auto work = [&] {
        while (!stop.load(std::memory_order_relaxed)) {
            const std::size_t begin = next.fetch_add(grain, std::memory_order_relaxed);
            if (begin >= n) {
                return;
            }
            const std::size_t end = std::min(begin + grain, n);
            try {
                for (std::size_t i = begin; i < end; ++i) {
                    fn(i);
                }
            } catch (...) {
                const std::lock_guard<std::mutex> lock(error_mutex);
                if (!error) {
                    error = std::current_exception();
                }
                stop = true;
                return;
            }
        }
    };

    if (workers <= 1) {
        work();
    } else {
        std::vector<std::thread> pool;
        pool.reserve(workers - 1);
        for (unsigned t = 1; t < workers; ++t) {
            pool.emplace_back(work);
        }
        work();
        for (std::thread &t : pool) {
            t.join();
        }
    }

    if (error) {
        std::rethrow_exception(error);
    }
}

} // namespace ppq

#endif
