#ifndef BENCH_BENCH_UTIL_H_
#define BENCH_BENCH_UTIL_H_

/// Shared by the microbenchmarks: the size presets, cached workloads and naming.
///
/// Benchmarks are registered at run time (after the preset is read) rather than with the
/// BENCHMARK macros, so one binary covers a full sweep and a seconds-long smoke run.

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "ppq/cpuref.h"
#include "ppq/generate.h"
#include "ppq/params.h"
#include "ppq/workload.h"

namespace bench {

/// --ppq_size: how far each sweep goes
enum class Size {
    /// One small case per benchmark: checks that everything runs (CTest, emulators)
    Smoke,
    /// The default: the shape of each sweep in a few minutes
    Quick,
    /// Every combination, up to the largest batches
    Full,
};

inline const char *size_name(Size s) {
    switch (s) {
    case Size::Smoke:
        return "smoke";
    case Size::Quick:
        return "quick";
    case Size::Full:
        return "full";
    }
    return "?";
}

/// Reads and removes --ppq_size=smoke|quick|full from argv, before benchmark::Initialize
inline Size take_size_flag(int &argc, char **argv) {
    Size size = Size::Quick;
    int out = 1;
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        constexpr const char *flag = "--ppq_size=";
        if (std::strncmp(a, flag, std::strlen(flag)) == 0) {
            const std::string v = a + std::strlen(flag);
            if (v == "smoke") {
                size = Size::Smoke;
            } else if (v == "quick") {
                size = Size::Quick;
            } else if (v == "full") {
                size = Size::Full;
            } else {
                std::cerr << "--ppq_size expects smoke, quick or full, got '" << v << "'\n";
                std::exit(2);
            }
            continue;
        }
        argv[out++] = argv[i];
    }
    argc = out;
    argv[argc] = nullptr;
    return size;
}

/// The sweep for a preset: `smoke` for Smoke, `quick` for Quick, `full` for Full
template <class T>
std::vector<T> pick(Size s, std::vector<T> smoke, std::vector<T> quick, std::vector<T> full) {
    switch (s) {
    case Size::Smoke:
        return smoke;
    case Size::Quick:
        return quick;
    case Size::Full:
        return full;
    }
    return quick;
}

inline std::vector<const ppq::ParamSet *> param_sets(Size s) {
    if (s == Size::Smoke) {
        return {&ppq::ML_DSA_44, &ppq::ML_DSA_87};
    }
    return {&ppq::ML_DSA_44, &ppq::ML_DSA_65, &ppq::ML_DSA_87};
}

inline const ppq::ParamSet &param_set(std::int64_t id) { return *ppq::find_param_set(int(id)); }

/// The fastest signer available: workloads only need to exist, not be signed a particular way
inline ppq::cpuref::Impl fastest_signer() {
    return ppq::cpuref::available(ppq::cpuref::Impl::Avx2) ? ppq::cpuref::Impl::Avx2
                                                           : ppq::cpuref::Impl::Ref;
}

/// All-valid external workload with 32-byte messages: signature i uses key i % keys
inline ppq::GenSpec bench_spec(const ppq::ParamSet &p, std::size_t sigs, std::size_t keys) {
    ppq::GenSpec s;
    s.params = p;
    s.signatures = sigs;
    s.keys = keys;
    s.seed = 0x5eed + static_cast<std::uint64_t>(p.id);
    s.impl = fastest_signer();
    return s;
}

/// The most recently requested workload, regenerated only when the request changes.
/// Benchmarks are registered grouped by workload so each one is signed once; one entry keeps
/// host memory bounded at the largest sizes.
inline const ppq::Workload &workload(const ppq::ParamSet &p, std::size_t sigs, std::size_t keys) {
    struct Entry {
        int params = 0;
        std::size_t sigs = 0;
        std::size_t keys = 0;
        std::unique_ptr<ppq::Workload> w;
    };
    static Entry cached;
    if (!cached.w || cached.params != p.id || cached.sigs != sigs || cached.keys != keys) {
        cached = Entry{}; // free the previous workload before signing the next
        cached.w = std::make_unique<ppq::Workload>(ppq::generate(bench_spec(p, sigs, keys)));
        cached.params = p.id;
        cached.sigs = sigs;
        cached.keys = keys;
    }
    return *cached.w;
}

/// Reported per benchmark so results carry their own units: signatures (or polynomials, keys,
/// hashes) per second, and bytes per second where bytes are meaningful
inline void set_items(benchmark::State &state, std::size_t items_per_iteration,
                      std::size_t bytes_per_iteration = 0) {
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) *
                            static_cast<std::int64_t>(items_per_iteration));
    if (bytes_per_iteration > 0) {
        state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) *
                                static_cast<std::int64_t>(bytes_per_iteration));
    }
}

/// Registers fn under `name`, forwarding the state
template <class Fn> ::benchmark::Benchmark *add(const std::string &name, Fn fn) {
    return benchmark::RegisterBenchmark(name.c_str(), fn);
}

inline std::string params_label(const ppq::ParamSet &p) { return std::to_string(p.id); }

inline void add_common_context(Size size) {
    benchmark::AddCustomContext("ppq_size", size_name(size));
    std::string impls;
    for (const ppq::cpuref::Impl impl : ppq::cpuref::available_impls()) {
        impls += (impls.empty() ? "" : ",") + std::string(ppq::cpuref::impl_name(impl));
    }
    benchmark::AddCustomContext("cpu_impls", impls);
    benchmark::AddCustomContext("hardware_threads",
                                std::to_string(std::thread::hardware_concurrency()));
}

} // namespace bench

#endif
