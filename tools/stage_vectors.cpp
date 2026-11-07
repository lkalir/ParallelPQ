/// Writes tests/vectors/stages/mldsa_stages.txt: every intermediate value of ML-DSA
/// verification for seeded workloads, computed with the pq-crystals reference internals.
///
/// usage: stage_vectors.exe OUTPUT
///
/// The output depends only on the constants below and the pinned pq-crystals commit, so
/// rerunning it reproduces the committed file; test_stages_host checks that it still does.

#include <array>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "ppq/cpuref_stages.h"
#include "ppq/drbg.h"
#include "ppq/generate.h"
#include "ppq/shake_vectors.h"

namespace {

using ppq::cpuref::Poly;

constexpr std::uint64_t STAGE_SEED = 0x5354414745; // "STAGE"
constexpr std::size_t KEYS_PER_SET = 2;
constexpr std::size_t SIGS_PER_SET = 10;
/// 4 of 10 pass; the 6 failures cycle through every tamper mode
constexpr double PASS_RATE = 0.4;
constexpr std::size_t RANDOM_NTT_VECTORS = 3;

class Writer {
  public:
    explicit Writer(std::ostream &out) : _out(out) {}

    void begin() { _out << "\n"; }
    void field(const char *key, const std::string &value) {
        _out << key << (value.empty() ? " =" : " = ") << value << "\n";
    }
    void field(const char *key, long long value) { field(key, std::to_string(value)); }
    void hex(const char *key, ppq::ByteView v) { field(key, ppq::to_hex(v.data(), v.size())); }

    void polys(const char *key, const std::vector<Poly> &ps) {
        _out << key << " =";
        for (const Poly &p : ps) {
            for (const std::int32_t c : p) {
                _out << ' ' << c;
            }
        }
        _out << "\n";
    }

    void poly(const char *key, const Poly &p) { polys(key, {p}); }

  private:
    std::ostream &_out;
};

void write_ntt_vectors(Writer &w) {
    std::vector<Poly> inputs;

    Poly delta{};
    delta[0] = 1;
    inputs.push_back(delta);

    Poly top{};
    top.fill(ppq::Q - 1);
    inputs.push_back(top);

    for (std::size_t v = 0; v < RANDOM_NTT_VECTORS; ++v) {
        ppq::Drbg rng(STAGE_SEED, "ntt", v);
        Poly a{};
        for (auto &c : a) {
            c = static_cast<std::int32_t>(rng.uniform(ppq::Q));
        }
        inputs.push_back(a);
    }

    for (const Poly &a : inputs) {
        const Poly a_hat = ppq::cpuref::ntt(a);
        if (ppq::cpuref::intt(a_hat) != a) {
            throw std::runtime_error("NTT round trip failed");
        }
        w.begin();
        w.field("kind", "ntt");
        w.poly("a", a);
        w.poly("a_hat", a_hat);
    }
}

void write_param_set(Writer &w, const ppq::ParamSet &p) {
    ppq::GenSpec spec;
    spec.params = p;
    spec.signatures = SIGS_PER_SET;
    spec.keys = KEYS_PER_SET;
    spec.pass_rate = PASS_RATE;
    spec.fail_modes = {ppq::Tamper::Message, ppq::Tamper::Commitment, ppq::Tamper::ZNorm,
                       ppq::Tamper::Hint};
    spec.msg_min = 0;
    spec.msg_max = 64;
    spec.ctx_max = 16;
    spec.seed = STAGE_SEED + static_cast<std::uint64_t>(p.id);
    const ppq::Workload wl = ppq::generate(spec);

    for (std::size_t k = 0; k < wl.key_count(); ++k) {
        const auto ks = ppq::cpuref::key_stages(p, wl.pk(k));
        w.begin();
        w.field("kind", "key");
        w.field("params", p.id);
        w.field("key", static_cast<long long>(k));
        w.hex("pk", wl.pk(k));
        w.hex("rho", ks.rho);
        w.polys("t1", ks.t1);
        w.hex("tr", ks.tr);
        w.polys("a_hat", ks.a_hat);
        w.polys("t1_hat", ks.t1_hat);
    }

    for (std::size_t i = 0; i < wl.sig_count(); ++i) {
        const auto mu = ppq::cpuref::mu_external(p, wl.pk_of(i), wl.msg(i), wl.ctx(i));
        const auto ss = ppq::cpuref::sig_stages(p, wl.pk_of(i), mu, wl.sig(i));

        if (ss.verdict != wl.expected[i]) {
            throw std::runtime_error(std::string(p.name) + " signature " + std::to_string(i) +
                                     ": pq-crystals says " + ppq::verdict_name(ss.verdict) +
                                     ", the generator expected " +
                                     ppq::verdict_name(wl.expected[i]));
        }

        w.begin();
        w.field("kind", "sig");
        w.field("params", p.id);
        w.field("key", static_cast<long long>(wl.key_index[i]));
        w.field("tamper", ppq::tamper_name(wl.tamper[i]));
        w.field("verdict", ppq::verdict_name(ss.verdict));
        w.hex("msg", wl.msg(i));
        w.hex("ctx", wl.ctx(i));
        w.hex("sig", wl.sig(i));
        w.hex("mu", mu);
        w.hex("ctilde", ss.ctilde);
        w.polys("z", ss.z);
        w.field("hint_ok", ss.hint_ok ? 1 : 0);
        if (ss.hint_ok) {
            w.polys("h", ss.h);
        }
        w.poly("c", ss.c);
        w.polys("z_hat", ss.z_hat);
        w.poly("c_hat", ss.c_hat);
        if (ss.hint_ok) {
            w.polys("w_approx", ss.w_approx);
            w.polys("w1", ss.w1);
            w.hex("w1_packed", ss.w1_packed);
            w.hex("ctilde_prime", ss.ctilde_prime);
        }
    }
}

} // namespace

int main(int argc, char **argv) try {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " OUTPUT\n";
        return 2;
    }

    const std::string path = argv[1];
    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        std::cerr << "error: cannot create '" << path << "'\n";
        return 1;
    }

    out << "# ML-DSA verification stages (FIPS 204 Algorithm 8) from the pq-crystals reference\n"
           "# generated by tools/stage_vectors.cpp (make stage-vectors); do not edit by hand\n"
           "#\n"
           "# kind = ntt: a and a_hat = NTT(a)\n"
           "# kind = key: pk, rho, t1, tr = H(pk), a_hat = ExpandA(rho), t1_hat = NTT(t1 * 2^d)\n"
           "# kind = sig: a signature on key `key` of the same parameter set, with mu, the\n"
           "#   decoded ctilde, z and h, c = SampleInBall(ctilde), z_hat, c_hat,\n"
           "#   w_approx = NTT^-1(A_hat o z_hat - c_hat o t1_hat), w1 = UseHint(h, w_approx),\n"
           "#   w1_packed = w1Encode(w1) and ctilde_prime = H(mu || w1_packed). When the hint\n"
           "#   is malformed (hint_ok = 0), h and everything from w_approx on are absent.\n"
           "#\n"
           "# Polynomial vectors are concatenated, row-major (a_hat[i][j] at (i * l + j) * 256).\n"
           "# Coefficients are in [0, q) except z (signed), c (-1, 0, 1) and h (0, 1).\n";

    Writer w(out);
    write_ntt_vectors(w);
    for (const ppq::ParamSet &p : ppq::PARAM_SETS) {
        write_param_set(w, p);
    }

    out.close();
    if (!out) {
        std::cerr << "error: writing '" << path << "' failed\n";
        return 1;
    }
    return 0;
} catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
}
