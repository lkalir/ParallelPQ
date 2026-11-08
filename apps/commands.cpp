#include "commands.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "ppq/acvp.h"
#include "ppq/cpu_verifier.h"
#include "ppq/parallel.h"

#ifndef PPQ_VERSION
#define PPQ_VERSION "unknown"
#endif

namespace ppq::app {
namespace {

using Clock = std::chrono::steady_clock;
/// Keeps keys in insertion order, so reports read top to bottom
using Json = nlohmann::ordered_json;

void print(std::ostream &os, const Json &j) { os << j.dump(2) << "\n"; }

double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::string fixed(double v, int decimals) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", decimals, v);
    return buf;
}

std::string ms(double v) { return fixed(v, v < 10.0 ? 3 : 1) + " ms"; }

/// 1234567 -> "1,234,567"
std::string grouped(double v) {
    std::string digits = fixed(v, 0);
    for (int i = static_cast<int>(digits.size()) - 3; i > 0; i -= 3) {
        digits.insert(static_cast<std::size_t>(i), ",");
    }
    return digits;
}

std::string impl_label(cpuref::Impl impl) {
    switch (impl) {
    case cpuref::Impl::Ref:
        return "pq-crystals ref";
    case cpuref::Impl::Avx2:
        return "pq-crystals AVX2";
    case cpuref::Impl::OpenSsl:
        return "OpenSSL";
    }
    return "unknown";
}

std::string gib(std::size_t bytes) {
    return fixed(static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0), 1) + " GiB";
}

/// "label   value" lines for the text reports
void line(std::ostream &os, const std::string &label, const std::string &value) {
    os << label << std::string(label.size() < 10 ? 10 - label.size() : 1, ' ') << value << "\n";
}

struct WorkloadSource {
    std::string kind; ///< "generated" or "file"
    std::string path;
    double generate_ms = 0.0;
    cpuref::Impl gen_impl = cpuref::Impl::Ref;
};

std::string counted(std::size_t n, const char *one, const char *many) {
    return grouped(static_cast<double>(n)) + " " + (n == 1 ? one : many);
}

std::string describe_workload(const Workload &w) {
    return std::string(w.params.name) + ", " + interface_name(w.iface) + ", " +
           counted(w.sig_count(), "signature", "signatures") + ", " +
           counted(w.key_count(), "key", "keys") + ", " +
           grouped(static_cast<double>(w.pass_count())) + " expected to pass";
}

Json workload_json(const Workload &w, const WorkloadSource &src) {
    Json j = {{"params", w.params.name},         {"interface", interface_name(w.iface)},
              {"signatures", w.sig_count()},     {"keys", w.key_count()},
              {"expected_pass", w.pass_count()}, {"source", src.kind}};
    if (src.kind == "file") {
        j["path"] = src.path;
    } else {
        j["seed"] = w.seed;
        j["gen_impl"] = cpuref::impl_name(src.gen_impl);
        j["generate_ms"] = src.generate_ms;
    }
    return j;
}

std::unique_ptr<Verifier> make_verifier(const Options &o, Interface iface) {
    CpuVerifierOptions v;
    v.impl = o.cpu_impl ? *o.cpu_impl : (o.verdicts ? cpuref::Impl::Ref : fastest_impl(iface));
    v.threads = o.threads;
    v.verdicts = o.verdicts;
    return std::make_unique<CpuVerifier>(v);
}

// ---------------------------------------------------------------------------------------- info

int cmd_info(const Options &o, Context &ctx) {
    const unsigned threads = resolve_threads(0);
    const auto impls = cpuref::available_impls();
    const GpuQuery gpus = ctx.query_gpus != nullptr ? ctx.query_gpus() : GpuQuery{};

    if (o.format == Format::Json) {
        Json root = {{"command", "info"},
                     {"version", PPQ_VERSION},
                     {"build", ctx.query_gpus != nullptr ? "gpu" : "cpu-only"}};
        if (ctx.query_gpus != nullptr) {
            root["gpu_platform"] = gpus.platform;
        }
        root["hardware_threads"] = threads;
        Json impl_items = Json::array();
        for (const cpuref::Impl impl : impls) {
            impl_items.push_back({{"name", cpuref::impl_name(impl)},
                                  {"description", impl_label(impl)},
                                  {"mu_interface", cpuref::supports(impl, Interface::Mu)}});
        }
        root["cpu_impls"] = std::move(impl_items);
        Json gpu_items = Json::array();
        for (const DeviceInfo &d : gpus.devices) {
            gpu_items.push_back({{"id", d.id},
                                 {"name", d.name},
                                 {"arch", d.arch},
                                 {"compute_units", d.compute_units},
                                 {"global_bytes", d.global_bytes}});
        }
        root["gpus"] = std::move(gpu_items);
        if (!gpus.error.empty()) {
            root["gpu_error"] = gpus.error;
        }
        print(ctx.out, root);
        return 0;
    }

    std::ostream &os = ctx.out;
    os << "ParallelPQ " << PPQ_VERSION << "\n";
    line(os, "build", ctx.query_gpus != nullptr ? "GPU (" + gpus.platform + ")" : "CPU only");
    line(os, "threads", std::to_string(threads) + " hardware threads");
    for (std::size_t i = 0; i < impls.size(); ++i) {
        std::string desc =
            std::string(cpuref::impl_name(impls[i])) + " (" + impl_label(impls[i]) + ")";
        if (!cpuref::supports(impls[i], Interface::Mu)) {
            desc += ", no mu interface";
        }
        line(os, i == 0 ? "cpu impls" : "", desc);
    }
    if (ctx.query_gpus == nullptr) {
        line(os, "gpus", "none: built without GPU support");
    } else if (!gpus.error.empty()) {
        line(os, "gpus", "none: " + gpus.error);
    } else if (gpus.devices.empty()) {
        line(os, "gpus", "none visible");
    } else {
        for (std::size_t i = 0; i < gpus.devices.size(); ++i) {
            const DeviceInfo &d = gpus.devices[i];
            line(os, i == 0 ? "gpus" : "",
                 std::to_string(d.id) + ": " + d.name + " (" + d.arch + ", " +
                     std::to_string(d.compute_units) + " CUs, " + gib(d.global_bytes) + ")");
        }
    }
    return 0;
}

// ------------------------------------------------------------------------------------ generate

int cmd_generate(const Options &o, Context &ctx) {
    const auto start = Clock::now();
    const Workload w = generate(o.spec);
    const double gen_ms = ms_since(start);
    save_workload(w, o.output);
    const auto bytes = std::filesystem::file_size(o.output);

    if (o.format == Format::Json) {
        WorkloadSource src{"generated", "", gen_ms, o.spec.impl};
        const Json root = {{"command", "generate"},
                           {"version", PPQ_VERSION},
                           {"output", o.output},
                           {"file_bytes", bytes},
                           {"workload", workload_json(w, src)}};
        print(ctx.out, root);
        return 0;
    }

    std::ostream &os = ctx.out;
    os << "ParallelPQ " << PPQ_VERSION << " generate\n";
    line(os, "workload", describe_workload(w));
    line(os, "signed",
         "seed " + std::to_string(w.seed) + ", " + impl_label(o.spec.impl) + ", " + ms(gen_ms));
    line(os, "wrote", o.output + " (" + grouped(static_cast<double>(bytes)) + " bytes)");
    return 0;
}

// -------------------------------------------------------------------------------------- verify

struct RunStats {
    std::vector<double> ms;
    double best = 0.0;
    double median = 0.0;
    double mean = 0.0;
};

RunStats stats(std::vector<double> times) {
    RunStats s;
    s.ms = times;
    std::sort(times.begin(), times.end());
    s.best = times.front();
    s.median = times.size() % 2 == 1
                   ? times[times.size() / 2]
                   : (times[times.size() / 2 - 1] + times[times.size() / 2]) / 2.0;
    s.mean = std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(times.size());
    return s;
}

int cmd_verify(const Options &o, Context &ctx) {
    WorkloadSource src;
    Workload w;
    if (!o.input.empty()) {
        src.kind = "file";
        src.path = o.input;
        w = load_workload(o.input);
    } else {
        src.kind = "generated";
        src.gen_impl = o.spec.impl;
        const auto start = Clock::now();
        w = generate(o.spec);
        src.generate_ms = ms_since(start);
    }

    const auto verifier = make_verifier(o, w.iface);
    if (!verifier->supports(w.iface)) {
        throw std::runtime_error(verifier->name() + " cannot verify " + interface_name(w.iface) +
                                 " workloads");
    }

    for (unsigned i = 0; i < o.warmup; ++i) {
        verifier->verify(w);
    }

    std::vector<double> times;
    VerifyResult result;
    Agreement agreement;
    bool consistent = true;
    for (unsigned i = 0; i < o.repeat; ++i) {
        VerifyResult r = verifier->verify(w);
        times.push_back(r.elapsed_ms);
        if (i > 0 && (r.valid != result.valid || r.verdicts != result.verdicts)) {
            consistent = false;
        }
        result = std::move(r);
    }
    agreement = compare(w, result);
    const RunStats t = stats(times);
    const double rate = t.best > 0.0 ? static_cast<double>(w.sig_count()) / (t.best / 1000.0) : 0.0;

    std::size_t per_verdict[VERDICT_COUNT] = {};
    for (const Verdict v : result.verdicts) {
        ++per_verdict[static_cast<std::size_t>(v)];
    }
    const bool ok = agreement.ok() && consistent;

    if (o.format == Format::Json) {
        Json res = {{"valid", agreement.valid}, {"invalid", w.sig_count() - agreement.valid}};
        if (!result.verdicts.empty()) {
            Json counts = Json::object();
            for (std::size_t v = 0; v < VERDICT_COUNT; ++v) {
                counts[verdict_name(static_cast<Verdict>(v))] = per_verdict[v];
            }
            res["verdicts"] = std::move(counts);
        }

        const Json root = {
            {"command", "verify"},
            {"version", PPQ_VERSION},
            {"workload", workload_json(w, src)},
            {"backend", {{"name", verifier->name()}, {"description", verifier->describe()}}},
            {"runs",
             {{"warmup", o.warmup},
              {"repeat", o.repeat},
              {"ms", t.ms},
              {"best_ms", t.best},
              {"median_ms", t.median},
              {"mean_ms", t.mean}}},
            {"sigs_per_second", rate},
            {"result", std::move(res)},
            {"check",
             {{"ok", ok},
              {"consistent_runs", consistent},
              {"wrong_validity", agreement.wrong_validity},
              {"wrong_verdict", agreement.wrong_verdict},
              {"examples", agreement.examples}}}};
        print(ctx.out, root);
        return ok ? 0 : 1;
    }

    std::ostream &os = ctx.out;
    os << "ParallelPQ " << PPQ_VERSION << " verify\n";
    line(os, "workload", describe_workload(w));
    line(os, "",
         src.kind == "file" ? "loaded from " + src.path
                            : "generated: seed " + std::to_string(w.seed) + ", " +
                                  impl_label(src.gen_impl) + ", " + ms(src.generate_ms));
    line(os, "backend", verifier->describe());
    std::string time = ms(t.best);
    if (o.repeat > 1) {
        time += " best, " + ms(t.median) + " median, " + ms(t.mean) + " mean over " +
                std::to_string(o.repeat) + " runs";
    }
    if (o.warmup > 0) {
        time += " (" + std::to_string(o.warmup) + " warmup)";
    }
    line(os, "time", time);
    line(os, "rate", grouped(rate) + " signatures/s");
    line(os, "result",
         grouped(static_cast<double>(agreement.valid)) + " valid, " +
             grouped(static_cast<double>(w.sig_count() - agreement.valid)) + " invalid");
    if (!result.verdicts.empty()) {
        std::string counts;
        for (std::size_t v = 0; v < VERDICT_COUNT; ++v) {
            counts += (v == 0 ? "" : ", ") + std::string(verdict_name(static_cast<Verdict>(v))) +
                      " " + std::to_string(per_verdict[v]);
        }
        line(os, "verdicts", counts);
    }

    if (ok) {
        line(os, "check",
             "all " + grouped(static_cast<double>(w.sig_count())) +
                 " agree with the expected verdicts");
        return 0;
    }
    if (!consistent) {
        line(os, "check", "FAILED: repeated runs gave different results");
    }
    if (!agreement.ok()) {
        line(os, "check",
             "FAILED: " + std::to_string(agreement.wrong_validity) + " wrong validity, " +
                 std::to_string(agreement.wrong_verdict) + " wrong verdicts");
        for (const std::size_t i : agreement.examples) {
            std::string got = result.valid[i] ? "valid" : "invalid";
            if (!result.verdicts.empty()) {
                got = verdict_name(result.verdicts[i]);
            }
            line(os, "",
                 "signature " + std::to_string(i) + ": expected " + verdict_name(w.expected[i]) +
                     ", got " + got);
        }
    }
    return 1;
}

// ----------------------------------------------------------------------------------------- kat

int cmd_kat(const Options &o, Context &ctx) {
    const std::string path = o.vectors_dir + "/acvp/mldsa_sigver.txt";
    const auto groups = load_acvp_sigver(path);

    struct GroupResult {
        const AcvpWorkload *group;
        std::string backend;
        bool skipped = false;
        Agreement agreement;
        double elapsed_ms = 0.0;
    };
    std::vector<GroupResult> results;
    std::size_t agree = 0;
    std::size_t disagree = 0;
    std::size_t skipped = 0;

    for (const AcvpWorkload &g : groups) {
        const Workload &w = g.workload;
        GroupResult gr{&g, "", false, {}, 0.0};
        // An explicit --cpu-impl may lack an interface; auto always picks one that has it
        const auto verifier = make_verifier(o, w.iface);
        gr.backend = verifier->name();
        if (!verifier->supports(w.iface)) {
            gr.skipped = true;
            skipped += w.sig_count();
        } else {
            const VerifyResult r = verifier->verify(w);
            gr.agreement = compare(w, r);
            gr.elapsed_ms = r.elapsed_ms;
            agree += w.sig_count() - gr.agreement.disagreeing;
            disagree += gr.agreement.disagreeing;
        }
        results.push_back(std::move(gr));
    }
    const bool ok = disagree == 0;

    if (o.format == Format::Json) {
        Json groups = Json::array();
        for (const GroupResult &gr : results) {
            const Workload &w = gr.group->workload;
            Json j = {{"params", w.params.name},
                      {"interface", interface_name(w.iface)},
                      {"vectors", w.sig_count()},
                      {"backend", gr.backend},
                      {"skipped", gr.skipped}};
            if (!gr.skipped) {
                Json bad = Json::array();
                for (const std::size_t i : gr.agreement.examples) {
                    bad.push_back(gr.group->tc_ids[i]);
                }
                j["ok"] = gr.agreement.ok();
                j["failed_tc_ids"] = std::move(bad);
                j["elapsed_ms"] = gr.elapsed_ms;
            }
            groups.push_back(std::move(j));
        }
        const Json root = {{"command", "kat"},   {"version", PPQ_VERSION},
                           {"vectors", path},    {"groups", std::move(groups)},
                           {"agree", agree},     {"disagree", disagree},
                           {"skipped", skipped}, {"ok", ok}};
        print(ctx.out, root);
        return ok ? 0 : 1;
    }

    std::ostream &os = ctx.out;
    os << "ParallelPQ " << PPQ_VERSION << " kat: NIST ACVP sigVer\n";
    line(os, "vectors", path);
    for (const GroupResult &gr : results) {
        const Workload &w = gr.group->workload;
        std::string label = std::string(w.params.name) + " " + interface_name(w.iface);
        label.resize(std::max<std::size_t>(label.size(), 19), ' ');
        std::string status;
        if (gr.skipped) {
            status = "skipped: " + gr.backend + " has no " + interface_name(w.iface) + " interface";
        } else if (gr.agreement.ok()) {
            status = "ok (" + gr.backend + ")";
        } else {
            status = "FAILED (" + gr.backend + "), tcId";
            for (const std::size_t i : gr.agreement.examples) {
                status += " " + std::to_string(gr.group->tc_ids[i]);
            }
        }
        line(os, "", label + " " + std::to_string(w.sig_count()) + " vectors  " + status);
    }
    line(os, "result",
         std::to_string(agree) + " agree, " + std::to_string(disagree) + " disagree, " +
             std::to_string(skipped) + " skipped");
    return ok ? 0 : 1;
}

} // namespace

int run_command(const Options &o, Context &ctx) {
    switch (o.command) {
    case Command::Help:
        ctx.out << (o.help_text.empty() ? usage_text() : o.help_text);
        return 0;
    case Command::Version:
        ctx.out << (o.version_text.empty() ? "ppq " PPQ_VERSION : o.version_text) << "\n";
        return 0;
    case Command::Info:
        return cmd_info(o, ctx);
    case Command::Generate:
        return cmd_generate(o, ctx);
    case Command::Verify:
        return cmd_verify(o, ctx);
    case Command::Kat:
        return cmd_kat(o, ctx);
    }
    return 1;
}

} // namespace ppq::app
