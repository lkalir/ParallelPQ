/// Host microbenchmarks: ParallelPQ's own host/device math (the PPQ_HD functions the kernels
/// call, compiled for the CPU) next to the pq-crystals equivalents, the CPU reference's
/// single-signature verify for every implementation, the multithreaded CPU verifier, and
/// workload generation and file I/O.
///
///   bench_host.exe [--ppq_size=smoke|quick|full] [Google Benchmark flags]

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "bench_util.h"
#include "ppq/cpu_verifier.h"
#include "ppq/cpuref_stages.h"
#include "ppq/decode.h"
#include "ppq/keccak.h"
#include "ppq/mldsa_traits.h"
#include "ppq/mu.h"
#include "ppq/ntt.h"
#include "ppq/rounding.h"
#include "ppq/sample.h"

namespace {

using bench::Size;
using benchmark::State;
using ppq::cpuref::Impl;

/// Signatures the single-signature benchmarks cycle through, so no one input stays hot
constexpr std::size_t SIG_POOL = 64;

std::array<std::int32_t, ppq::N> random_poly(std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::int32_t> coeff(0, ppq::Q - 1);
    std::array<std::int32_t, ppq::N> a{};
    for (std::int32_t &c : a) {
        c = coeff(rng);
    }
    return a;
}

std::vector<std::uint8_t> random_bytes(std::size_t n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<std::uint8_t> out(n);
    for (std::uint8_t &b : out) {
        b = static_cast<std::uint8_t>(rng());
    }
    return out;
}

// ----------------------------------------------------------------------------------- hashing

template <class Xof> void shake_ppq(State &state, std::size_t in_len) {
    const auto in = random_bytes(in_len, 1);
    std::uint8_t out[64];
    for (auto _ : state) {
        ppq::shake<Xof>(in.data(), in.size(), out, sizeof(out));
        benchmark::DoNotOptimize(out);
        benchmark::ClobberMemory();
    }
    bench::set_items(state, 1, in_len);
}

void register_hashing(Size size) {
    for (const std::size_t in_len : bench::pick<std::size_t>(size, {32}, {32, 1312, 2592},
                                                             {32, 168, 1312, 1952, 2592, 4627})) {
        const std::string len = "/in_bytes:" + std::to_string(in_len);
        bench::add("shake128/ppq" + len, [=](State &s) { shake_ppq<ppq::Shake128>(s, in_len); });
        bench::add("shake256/ppq" + len, [=](State &s) { shake_ppq<ppq::Shake256>(s, in_len); });
    }

    // tr and mu for an external signature: the verifier's first hashes
    for (const ppq::ParamSet *p : bench::param_sets(size)) {
        const std::string tag = "/params:" + bench::params_label(*p);
        bench::add("mu_external/ppq" + tag, [p](State &state) {
            const ppq::Workload &w = bench::workload(*p, SIG_POOL, SIG_POOL);
            std::uint8_t tr[ppq::TR_BYTES];
            std::uint8_t mu[ppq::MU_BYTES];
            std::size_t i = 0;
            for (auto _ : state) {
                const auto pk = w.pk_of(i);
                const auto msg = w.msg(i);
                const auto ctx = w.ctx(i);
                ppq::compute_tr(pk.data(), pk.size(), tr);
                ppq::compute_mu(ppq::Interface::External, tr, msg.data(), msg.size(), ctx.data(),
                                ctx.size(), mu);
                benchmark::DoNotOptimize(mu);
                i = (i + 1) % SIG_POOL;
            }
            bench::set_items(state, 1);
        });
        bench::add("mu_external/pqcrystals" + tag, [p](State &state) {
            const ppq::Workload &w = bench::workload(*p, SIG_POOL, SIG_POOL);
            std::size_t i = 0;
            for (auto _ : state) {
                auto mu = ppq::cpuref::mu_external(*p, w.pk_of(i), w.msg(i), w.ctx(i));
                benchmark::DoNotOptimize(mu.data());
                i = (i + 1) % SIG_POOL;
            }
            bench::set_items(state, 1);
        });
    }
}

// --------------------------------------------------------------------------------------- NTT

void register_ntt() {
    // Each iteration transforms a fresh copy (the transforms are in place and grow the
    // coefficients), and so does pq-crystals' by-value interface: the copy is in both.
    const auto input = random_poly(7);
    bench::add("ntt/ppq/dir:forward", [=](State &state) {
        for (auto _ : state) {
            auto a = input;
            ppq::ntt(a.data());
            benchmark::DoNotOptimize(a.data());
        }
        bench::set_items(state, 1);
    });
    bench::add("ntt/ppq/dir:inverse", [=](State &state) {
        for (auto _ : state) {
            auto a = input;
            ppq::invntt_tomont(a.data());
            benchmark::DoNotOptimize(a.data());
        }
        bench::set_items(state, 1);
    });
    // The FIPS 204 wrappers over the pq-crystals ref transforms: canonical in and out, so
    // they include the conversion to and from Montgomery form
    bench::add("ntt/pqcrystals/dir:forward", [=](State &state) {
        for (auto _ : state) {
            auto a = ppq::cpuref::ntt(input);
            benchmark::DoNotOptimize(a.data());
        }
        bench::set_items(state, 1);
    });
    bench::add("ntt/pqcrystals/dir:inverse", [=](State &state) {
        for (auto _ : state) {
            auto a = ppq::cpuref::intt(input);
            benchmark::DoNotOptimize(a.data());
        }
        bench::set_items(state, 1);
    });
}

// ------------------------------------------------------------------- sampling and decoding

void register_sampling(Size size) {
    const auto rho = random_bytes(ppq::SEED_BYTES, 3);
    bench::add("expand_a_poly/ppq", [=](State &state) {
        std::int32_t a[ppq::N];
        unsigned entry = 0;
        for (auto _ : state) {
            ppq::expand_a_poly(rho.data(), entry / 8, entry % 8, a);
            benchmark::DoNotOptimize(a);
            entry = (entry + 1) % 64;
        }
        bench::set_items(state, 1);
    });

    for (const ppq::ParamSet *p : bench::param_sets(size)) {
        const std::string tag = "/params:" + bench::params_label(*p);
        ppq::with_params(p->id, [&](auto m) {
            using M = decltype(m);
            constexpr int P = M::id;

            bench::add("sample_in_ball/ppq" + tag, [p](State &state) {
                const ppq::Workload &w = bench::workload(*p, SIG_POOL, SIG_POOL);
                std::int32_t c[ppq::N];
                std::size_t i = 0;
                for (auto _ : state) {
                    ppq::sample_in_ball<P>(w.sig(i).data(), c);
                    benchmark::DoNotOptimize(c);
                    i = (i + 1) % SIG_POOL;
                }
                bench::set_items(state, 1);
            });

            // z for one signature: l polynomials decoded and norm checked
            bench::add("z_decode/ppq" + tag, [p](State &state) {
                const ppq::Workload &w = bench::workload(*p, SIG_POOL, SIG_POOL);
                std::int32_t z[ppq::N];
                std::size_t i = 0;
                for (auto _ : state) {
                    const std::uint8_t *zb = w.sig(i).data() + M::ctilde_bytes;
                    bool ok = true;
                    for (int j = 0; j < M::l; ++j) {
                        ppq::z_decode_poly<P>(zb + j * M::z_poly_bytes, z);
                        ok = ppq::z_poly_norm_ok<P>(z) && ok;
                    }
                    benchmark::DoNotOptimize(ok);
                    benchmark::DoNotOptimize(z);
                    i = (i + 1) % SIG_POOL;
                }
                bench::set_items(state, 1);
            });

            bench::add("hint_decode/ppq" + tag, [p](State &state) {
                const ppq::Workload &w = bench::workload(*p, SIG_POOL, SIG_POOL);
                std::vector<std::uint8_t> h(static_cast<std::size_t>(M::k) * ppq::N);
                std::size_t i = 0;
                for (auto _ : state) {
                    const std::uint8_t *y =
                        w.sig(i).data() + M::ctilde_bytes + M::l * M::z_poly_bytes;
                    benchmark::DoNotOptimize(ppq::hint_decode<P>(y, h.data()));
                    benchmark::DoNotOptimize(h.data());
                    i = (i + 1) % SIG_POOL;
                }
                bench::set_items(state, 1);
            });

            // One row of w1: UseHint on 256 coefficients, then w1Encode
            bench::add("use_hint_row/ppq" + tag, [](State &state) {
                const auto r = random_poly(11);
                const auto hint = random_bytes(ppq::N, 12);
                std::int32_t w1[ppq::N];
                std::uint8_t packed[M::w1_poly_bytes];
                for (auto _ : state) {
                    for (unsigned c = 0; c < ppq::N; ++c) {
                        w1[c] = ppq::use_hint<P>(hint[c] & 1, r[c]);
                    }
                    ppq::w1_pack_poly<P>(w1, packed);
                    benchmark::DoNotOptimize(packed);
                }
                bench::set_items(state, 1);
            });
        });
    }

    bench::add("t1_decode_poly/ppq", [](State &state) {
        const auto packed = random_bytes(ppq::T1_PACKED_BYTES, 5);
        std::int32_t t1[ppq::N];
        for (auto _ : state) {
            ppq::t1_decode_poly(packed.data(), t1);
            benchmark::DoNotOptimize(t1);
        }
        bench::set_items(state, 1);
    });
}

// ------------------------------------------------------------------ pq-crystals stage oracle

void register_stage_oracle(Size size) {
    for (const ppq::ParamSet *p : bench::param_sets(size)) {
        const std::string tag = "/params:" + bench::params_label(*p);
        bench::add("key_stages/pqcrystals" + tag, [p](State &state) {
            const ppq::Workload &w = bench::workload(*p, SIG_POOL, SIG_POOL);
            std::size_t i = 0;
            for (auto _ : state) {
                auto k = ppq::cpuref::key_stages(*p, w.pk(i));
                benchmark::DoNotOptimize(k.a_hat.data());
                i = (i + 1) % SIG_POOL;
            }
            bench::set_items(state, 1);
        });
        bench::add("sig_stages/pqcrystals" + tag, [p](State &state) {
            const ppq::Workload &w = bench::workload(*p, SIG_POOL, SIG_POOL);
            std::vector<ppq::cpuref::Bytes> mus;
            for (std::size_t i = 0; i < SIG_POOL; ++i) {
                mus.push_back(ppq::cpuref::mu_external(*p, w.pk_of(i), w.msg(i), w.ctx(i)));
            }
            std::size_t i = 0;
            for (auto _ : state) {
                auto s = ppq::cpuref::sig_stages(*p, w.pk_of(i), mus[i], w.sig(i));
                benchmark::DoNotOptimize(s.ctilde_prime.data());
                i = (i + 1) % SIG_POOL;
            }
            bench::set_items(state, 1);
        });
    }
}

// ------------------------------------------------------------------- CPU reference verify

void register_cpuref(Size size) {
    for (const ppq::ParamSet *p : bench::param_sets(size)) {
        for (const Impl impl : ppq::cpuref::available_impls()) {
            const std::string tag = std::string("/impl:") + ppq::cpuref::impl_name(impl) +
                                    "/params:" + bench::params_label(*p);
            bench::add("cpuref_verify" + tag, [p, impl](State &state) {
                const ppq::Workload &w = bench::workload(*p, SIG_POOL, SIG_POOL);
                std::size_t i = 0;
                for (auto _ : state) {
                    benchmark::DoNotOptimize(
                        ppq::cpuref::verify(*p, w.pk_of(i), w.msg(i), w.ctx(i), w.sig(i), impl));
                    i = (i + 1) % SIG_POOL;
                }
                bench::set_items(state, 1);
            });
            if (!ppq::cpuref::supports(impl, ppq::Interface::Mu)) {
                continue;
            }
            // From a precomputed mu: the verification work without the message hash
            bench::add("cpuref_verify_mu" + tag, [p, impl](State &state) {
                const ppq::Workload &w = bench::workload(*p, SIG_POOL, SIG_POOL);
                std::vector<ppq::cpuref::Bytes> mus;
                for (std::size_t i = 0; i < SIG_POOL; ++i) {
                    mus.push_back(ppq::cpuref::mu_external(*p, w.pk_of(i), w.msg(i), w.ctx(i)));
                }
                std::size_t i = 0;
                for (auto _ : state) {
                    benchmark::DoNotOptimize(
                        ppq::cpuref::verify_mu(*p, w.pk_of(i), mus[i], w.sig(i), impl));
                    i = (i + 1) % SIG_POOL;
                }
                bench::set_items(state, 1);
            });
        }
    }
}

// -------------------------------------------------------------- multithreaded CPU verifier

std::vector<unsigned> thread_counts(Size size) {
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    if (size == Size::Smoke) {
        return {std::min(2u, hw)};
    }
    std::vector<unsigned> out;
    for (unsigned t = 1; t < hw; t *= 2) {
        out.push_back(t);
    }
    out.push_back(hw);
    return out;
}

void register_cpu_verifier(Size size) {
    const std::size_t sigs = bench::pick<std::size_t>(size, {64}, {4096}, {16384})[0];
    for (const ppq::ParamSet *p : bench::param_sets(size)) {
        for (const unsigned threads : thread_counts(size)) {
            for (const bool verdicts : {false, true}) {
                ppq::CpuVerifierOptions o;
                o.impl = verdicts ? Impl::Ref : bench::fastest_signer();
                o.threads = threads;
                o.verdicts = verdicts;
                const std::string tag =
                    std::string("/impl:") + ppq::cpuref::impl_name(o.impl) +
                    (verdicts ? "-verdicts" : "") + "/params:" + bench::params_label(*p) +
                    "/sigs:" + std::to_string(sigs) + "/threads:" + std::to_string(threads);
                bench::add("cpu_verifier" + tag,
                           [p, o, sigs](State &state) {
                               const ppq::Workload &w = bench::workload(*p, sigs, sigs);
                               ppq::CpuVerifier v(o);
                               for (auto _ : state) {
                                   auto r = v.verify(w);
                                   benchmark::DoNotOptimize(r.valid.data());
                               }
                               bench::set_items(state, sigs);
                           })
                    ->UseRealTime()
                    ->Unit(benchmark::kMillisecond);
            }
        }
    }
}

// ----------------------------------------------------------------- workloads and .ppq files

void register_workloads(Size size) {
    const std::size_t sign_sigs = bench::pick<std::size_t>(size, {16}, {256}, {1024})[0];
    const std::size_t file_sigs = bench::pick<std::size_t>(size, {64}, {4096}, {16384})[0];
    for (const ppq::ParamSet *p : bench::param_sets(size)) {
        const std::string tag = "/params:" + bench::params_label(*p);
        // Key generation and signing on every hardware thread with the fastest signer
        bench::add("generate" + tag + "/sigs:" + std::to_string(sign_sigs),
                   [p, sign_sigs](State &state) {
                       const ppq::GenSpec spec = bench::bench_spec(*p, sign_sigs, sign_sigs);
                       for (auto _ : state) {
                           auto w = ppq::generate(spec);
                           benchmark::DoNotOptimize(w.sigs.data());
                       }
                       bench::set_items(state, sign_sigs);
                   })
            ->UseRealTime()
            ->Unit(benchmark::kMillisecond);

        const std::string path = (std::filesystem::temp_directory_path() /
                                  ("ppq_bench_" + std::to_string(p->id) + ".ppq"))
                                     .string();
        bench::add("workload_save" + tag + "/sigs:" + std::to_string(file_sigs),
                   [p, file_sigs, path](State &state) {
                       const ppq::Workload &w = bench::workload(*p, file_sigs, file_sigs);
                       for (auto _ : state) {
                           ppq::save_workload(w, path);
                       }
                       bench::set_items(state, file_sigs, std::filesystem::file_size(path));
                       std::filesystem::remove(path);
                   })
            ->UseRealTime()
            ->Unit(benchmark::kMillisecond);
        bench::add("workload_load" + tag + "/sigs:" + std::to_string(file_sigs),
                   [p, file_sigs, path](State &state) {
                       ppq::save_workload(bench::workload(*p, file_sigs, file_sigs), path);
                       for (auto _ : state) {
                           auto w = ppq::load_workload(path);
                           benchmark::DoNotOptimize(w.sigs.data());
                       }
                       bench::set_items(state, file_sigs, std::filesystem::file_size(path));
                       std::filesystem::remove(path);
                   })
            ->UseRealTime()
            ->Unit(benchmark::kMillisecond);
    }
}

} // namespace

int main(int argc, char **argv) {
    const Size size = bench::take_size_flag(argc, argv);
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 2;
    }
    bench::add_common_context(size);

    register_hashing(size);
    register_ntt();
    register_sampling(size);
    register_stage_oracle(size);
    register_cpuref(size);
    register_cpu_verifier(size);
    register_workloads(size);

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
