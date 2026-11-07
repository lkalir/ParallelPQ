/// tests/vectors/stages/mldsa_stages.txt against two independent sources: the naive FIPS 204
/// transcription in naive_mldsa.h, and a fresh run of the pinned pq-crystals internals.

#include <map>
#include <set>
#include <string>
#include <utility>

#include "check.h"
#include "naive_mldsa.h"
#include "ppq/cpuref_stages.h"
#include "vector_util.h"

namespace {

using vector_util::params_of;
using vector_util::poly;
using vector_util::polys;

const std::vector<ppq::KvRecord> &records() {
    static const std::vector<ppq::KvRecord> r =
        ppq::load_kv_records(TEST_VECTORS_DIR "/stages/mldsa_stages.txt");
    return r;
}

std::vector<const ppq::KvRecord *> of_kind(const std::string &kind) {
    std::vector<const ppq::KvRecord *> out;
    for (const ppq::KvRecord &r : records()) {
        if (r.str("kind") == kind) {
            out.push_back(&r);
        }
    }
    return out;
}

/// (params, key index) -> pk
std::map<std::pair<int, std::int64_t>, naive::Bytes> keys() {
    std::map<std::pair<int, std::int64_t>, naive::Bytes> out;
    for (const ppq::KvRecord *r : of_kind("key")) {
        out[{params_of(*r).id, r->integer("key")}] = r->bytes("pk");
    }
    return out;
}

const naive::Bytes &pk_for(const ppq::KvRecord &sig) {
    static const auto k = keys();
    const auto it = k.find({params_of(sig).id, sig.integer("key")});
    if (it == k.end()) {
        throw std::runtime_error(sig.where() + ": no such key");
    }
    return it->second;
}

std::size_t ku(const ppq::ParamSet &p) { return static_cast<std::size_t>(p.k); }
std::size_t lu(const ppq::ParamSet &p) { return static_cast<std::size_t>(p.l); }

TEST_CASE(corpus_covers_every_parameter_set_and_verdict) {
    CHECK(of_kind("ntt").size() >= 5);
    for (const ppq::ParamSet &p : ppq::PARAM_SETS) {
        std::size_t key_count = 0;
        std::set<ppq::Verdict> verdicts;
        for (const ppq::KvRecord *r : of_kind("key")) {
            key_count += params_of(*r).id == p.id;
        }
        for (const ppq::KvRecord *r : of_kind("sig")) {
            if (params_of(*r).id == p.id) {
                verdicts.insert(vector_util::parse_verdict(*r));
            }
        }
        CHECK(key_count == 2);
        CHECK(verdicts.size() == ppq::VERDICT_COUNT);
    }
}

TEST_CASE(ntt_vectors_match_the_definition) {
    for (const ppq::KvRecord *r : of_kind("ntt")) {
        const naive::Poly a = poly(*r, "a");
        const naive::Poly a_hat = poly(*r, "a_hat");
        CHECK(naive::ntt(a) == a_hat);
        CHECK(naive::intt(a_hat) == a);
        CHECK(ppq::cpuref::ntt(a) == a_hat);
        CHECK(ppq::cpuref::intt(a_hat) == a);
    }
}

TEST_CASE(key_stages_match_naive_math) {
    for (const ppq::KvRecord *r : of_kind("key")) {
        const ppq::ParamSet &p = params_of(*r);
        const naive::Bytes pk = r->bytes("pk");
        const naive::Stages n =
            naive::verify_stages(p, pk, naive::Bytes(64), naive::Bytes(p.sig_bytes));

        CHECK(r->bytes("rho") == n.rho);
        CHECK(polys(*r, "t1", ku(p)) == n.t1);
        CHECK(r->bytes("tr") == n.tr);
        CHECK(polys(*r, "a_hat", ku(p) * lu(p)) == n.a_hat);
        CHECK(polys(*r, "t1_hat", ku(p)) == n.t1_hat);
    }
}

TEST_CASE(sig_stages_match_naive_math) {
    for (const ppq::KvRecord *r : of_kind("sig")) {
        const ppq::ParamSet &p = params_of(*r);
        const naive::Bytes &pk = pk_for(*r);
        const naive::Bytes mu = r->bytes("mu");
        const naive::Bytes sig = r->bytes("sig");
        const ppq::Verdict verdict = vector_util::parse_verdict(*r);

        CHECK(naive::mu_external(pk, r->bytes("msg"), r->bytes("ctx")) == mu);

        const naive::Stages n = naive::verify_stages(p, pk, mu, sig);
        CHECK(n.verdict == verdict);
        CHECK(ppq::expected_verdict(*ppq::parse_tamper(r->str("tamper"))) == verdict);
        CHECK(r->bytes("ctilde") == n.ctilde);
        CHECK(polys(*r, "z", lu(p)) == n.z);
        CHECK(poly(*r, "c") == n.c);
        CHECK(polys(*r, "z_hat", lu(p)) == n.z_hat);
        CHECK(poly(*r, "c_hat") == n.c_hat);

        // SampleInBall puts exactly tau nonzero coefficients, all +-1
        std::size_t weight = 0;
        for (const std::int32_t c : n.c) {
            weight += c != 0;
            CHECK(c >= -1 && c <= 1);
        }
        CHECK(weight == static_cast<std::size_t>(p.tau));

        CHECK((r->integer("hint_ok") == 1) == n.h.has_value());
        if (n.h) {
            CHECK(polys(*r, "h", ku(p)) == *n.h);
            CHECK(polys(*r, "w_approx", ku(p)) == n.w_approx);
            CHECK(polys(*r, "w1", ku(p)) == n.w1);
            CHECK(r->bytes("w1_packed") == n.w1_packed);
            CHECK(r->bytes("ctilde_prime") == n.ctilde_prime);
        } else {
            CHECK(!r->has("h") && !r->has("w_approx") && !r->has("ctilde_prime"));
        }
    }
}

/// The committed file is exactly what the pinned pq-crystals commit produces today
TEST_CASE(corpus_is_reproducible_from_pq_crystals) {
    for (const ppq::KvRecord *r : of_kind("key")) {
        const ppq::ParamSet &p = params_of(*r);
        const auto ks = ppq::cpuref::key_stages(p, r->bytes("pk"));
        CHECK(polys(*r, "a_hat", ku(p) * lu(p)) == ks.a_hat);
        CHECK(polys(*r, "t1_hat", ku(p)) == ks.t1_hat);
        CHECK(r->bytes("tr") == ks.tr);
    }

    for (const ppq::KvRecord *r : of_kind("sig")) {
        const ppq::ParamSet &p = params_of(*r);
        const naive::Bytes &pk = pk_for(*r);
        const naive::Bytes mu = ppq::cpuref::mu_external(p, pk, r->bytes("msg"), r->bytes("ctx"));
        CHECK(mu == r->bytes("mu"));

        const auto ss = ppq::cpuref::sig_stages(p, pk, mu, r->bytes("sig"));
        CHECK(ss.verdict == vector_util::parse_verdict(*r));
        CHECK(polys(*r, "z_hat", lu(p)) == ss.z_hat);
        if (ss.hint_ok) {
            CHECK(polys(*r, "w_approx", ku(p)) == ss.w_approx);
            CHECK(r->bytes("ctilde_prime") == ss.ctilde_prime);
        }
    }
}

} // namespace

int main() { return check::run_all(); }
