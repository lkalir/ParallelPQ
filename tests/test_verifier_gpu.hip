/// The staged GPU verifier end to end: ACVP vectors and generated workloads, with its results and
/// reasons checked against the expected verdicts and the CPU reference

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

#include "gpu_test.h"
#include "ppq/acvp.h"
#include "ppq/cpu_verifier.h"
#include "ppq/generate.h"
#include "ppq/gpu_verifier.h"

namespace {

ppq::GenSpec spec(const ppq::ParamSet &p, std::size_t n, std::size_t keys, double pass_rate,
                  std::uint64_t seed) {
    ppq::GenSpec s;
    s.params = p;
    s.signatures = n;
    s.keys = keys;
    s.pass_rate = pass_rate;
    s.fail_modes = {ppq::Tamper::Message, ppq::Tamper::Commitment, ppq::Tamper::ZNorm,
                    ppq::Tamper::Hint};
    s.msg_min = 0;
    s.msg_max = 80;
    s.ctx_max = 20;
    s.seed = seed;
    s.threads = 2;
    return s;
}

/// Verifies on the GPU and checks validity and reasons against the expectations and the CPU
bool verifies_correctly(ppq::GpuVerifier &gpu, const ppq::Workload &w, const std::string &what) {
    const ppq::VerifyResult r = gpu.verify(w);
    const ppq::Agreement a = ppq::compare(w, r);
    ppq::CpuVerifier cpu({ppq::cpuref::Impl::Ref, 2, true});
    const bool same_as_cpu = r.verdicts == cpu.verify(w).verdicts;
    if (!a.ok() || !same_as_cpu) {
        std::cerr << "    " << what << ": " << a.wrong_validity << " wrong validity, "
                  << a.wrong_verdict << " wrong verdicts";
        for (const std::size_t i : a.examples) {
            std::cerr << " | #" << i << " expected " << ppq::verdict_name(w.expected[i]) << " got "
                      << ppq::verdict_name(r.verdicts[i]);
        }
        std::cerr << "\n";
    }
    return a.ok() && same_as_cpu;
}

TEST_CASE(acvp_every_interface) {
    ppq::GpuVerifier gpu;
    for (const ppq::AcvpWorkload &g :
         ppq::load_acvp_sigver(TEST_VECTORS_DIR "/acvp/mldsa_sigver.txt")) {
        const std::string what =
            std::string(g.workload.params.name) + " " + ppq::interface_name(g.workload.iface);
        CHECK(verifies_correctly(gpu, g.workload, what));
    }
}

TEST_CASE(distributions_for_every_parameter_set) {
    ppq::GpuVerifier gpu;
    for (const ppq::ParamSet &p : ppq::PARAM_SETS) {
        const std::uint64_t seed = 2000 + static_cast<std::uint64_t>(p.id);
        CHECK(verifies_correctly(gpu, ppq::generate(spec(p, 24, 1, 1.0, seed)),
                                 "same key, all pass"));
        CHECK(verifies_correctly(gpu, ppq::generate(spec(p, 24, 0, 0.5, seed + 1)),
                                 "distinct keys, half pass"));
        CHECK(verifies_correctly(gpu, ppq::generate(spec(p, 24, 4, 0.0, seed + 2)), "none pass"));
    }
}

TEST_CASE(chunking_and_key_remapping) {
    // 40 signatures in chunks of 7 (the last one partial), with key tables that fit the cache
    // (3 keys), exceed it with repeats (20 keys), and give every signature its own key
    ppq::GpuVerifierOptions o;
    o.chunk = 7;
    ppq::GpuVerifier gpu(o);
    for (const std::size_t keys : {3u, 20u, 0u}) {
        const ppq::Workload w = ppq::generate(spec(ppq::ML_DSA_65, 40, keys, 0.5, 2100 + keys));
        CHECK(verifies_correctly(gpu, w, "chunks of 7, " + std::to_string(keys) + " keys"));
    }
}

TEST_CASE(launch_geometries) {
    const ppq::Workload w = ppq::generate(spec(ppq::ML_DSA_44, 20, 2, 0.5, 2200));
    for (const auto &[threads, group] :
         {std::pair{64u, 16u}, std::pair{128u, 32u}, std::pair{256u, 64u}, std::pair{128u, 128u}}) {
        ppq::GpuVerifierOptions o;
        o.threads = threads;
        o.ntt_group = group;
        ppq::GpuVerifier gpu(o);
        CHECK(verifies_correctly(gpu, w, std::to_string(threads) + "/" + std::to_string(group)));
    }
}

TEST_CASE(buffers_are_reused_and_resized) {
    ppq::GpuVerifier gpu;
    const ppq::Workload small = ppq::generate(spec(ppq::ML_DSA_44, 4, 1, 0.5, 2300));
    const ppq::Workload big = ppq::generate(spec(ppq::ML_DSA_87, 12, 3, 0.5, 2301));
    CHECK(verifies_correctly(gpu, small, "small 44"));
    CHECK(verifies_correctly(gpu, big, "bigger 87"));
    CHECK(verifies_correctly(gpu, small, "small 44 again"));
    CHECK(verifies_correctly(gpu, big, "bigger 87 again"));
}

TEST_CASE(edge_cases_and_reporting) {
    ppq::GpuVerifier gpu;
    ppq::Workload empty;
    const ppq::VerifyResult none = gpu.verify(empty);
    CHECK(none.valid.empty() && none.verdicts.empty());

    const ppq::Workload one = ppq::generate(spec(ppq::ML_DSA_65, 1, 1, 1.0, 2400));
    const ppq::VerifyResult r = gpu.verify(one);
    CHECK(r.valid.size() == 1 && r.valid[0] == 1);
    CHECK(r.elapsed_ms > 0.0);

    // Every stage reports a time
    for (const char *stage : {"upload", "keys", "decode", "mu", "challenge", "ntt_z", "matvec",
                              "intt_w", "use_hint", "final_hash", "download"}) {
        bool found = false;
        for (const ppq::StageTime &t : r.stages) {
            found = found || (t.name == stage && t.ms >= 0.0);
        }
        CHECK(found);
    }
    CHECK(gpu.reports_verdicts());
    CHECK(gpu.supports(ppq::Interface::Mu));
    CHECK(gpu.describe().find("staged") != std::string::npos);
}

TEST_CASE(bad_options_are_rejected) {
    const auto rejects = [](unsigned threads, unsigned group, std::size_t chunk) {
        ppq::GpuVerifierOptions o;
        o.threads = threads;
        o.ntt_group = group;
        o.chunk = chunk;
        try {
            ppq::GpuVerifier gpu(o);
        } catch (const std::invalid_argument &) {
            return true;
        }
        return false;
    };
    CHECK(rejects(96, 64, 16));  // not a multiple of the group
    CHECK(rejects(128, 48, 16)); // unsupported group
    CHECK(rejects(0, 32, 16));
    CHECK(rejects(128, 32, 0));
    CHECK(!rejects(128, 32, 16));
}

} // namespace

int main() { return gpu_test::run_all(); }
