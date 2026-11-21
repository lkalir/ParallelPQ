/// tr and mu (ppq/mu.h) on the host: stage, ACVP and module 6 hashlib vectors, and the naive
/// transcription across block boundaries

#include "check.h"
#include "mldsa_vectors.h"
#include "naive_mldsa.h"
#include "ppq/acvp.h"
#include "ppq/cpuref_stages.h"
#include "ppq/drbg.h"
#include "ppq/mu.h"
#include "vector_util.h"

namespace {

using naive::Bytes;

Bytes tr_of(const Bytes &pk) {
    Bytes tr(ppq::TR_BYTES);
    ppq::compute_tr(pk.data(), pk.size(), tr.data());
    return tr;
}

Bytes mu_of(ppq::Interface iface, const Bytes &tr, const Bytes &msg, const Bytes &ctx) {
    Bytes mu(ppq::MU_BYTES);
    ppq::compute_mu(iface, tr.data(), msg.data(), msg.size(), ctx.data(), ctx.size(), mu.data());
    return mu;
}

TEST_CASE(stage_vectors) {
    std::map<std::pair<int, std::int64_t>, Bytes> pks;
    std::size_t sigs = 0;
    for (const ppq::KvRecord &r :
         ppq::load_kv_records(TEST_VECTORS_DIR "/stages/mldsa_stages.txt")) {
        if (r.str("kind") == "key") {
            CHECK(tr_of(r.bytes("pk")) == r.bytes("tr"));
            pks[{vector_util::params_of(r).id, r.integer("key")}] = r.bytes("pk");
        } else if (r.str("kind") == "sig") {
            const Bytes &pk = pks.at({vector_util::params_of(r).id, r.integer("key")});
            CHECK(mu_of(ppq::Interface::External, tr_of(pk), r.bytes("msg"), r.bytes("ctx")) ==
                  r.bytes("mu"));
            ++sigs;
        }
    }
    CHECK(sigs == 30);
}

TEST_CASE(acvp_vectors_every_interface) {
    for (const ppq::AcvpWorkload &g :
         ppq::load_acvp_sigver(TEST_VECTORS_DIR "/acvp/mldsa_sigver.txt")) {
        const ppq::Workload &w = g.workload;
        for (std::size_t i = 0; i < w.sig_count(); ++i) {
            const Bytes pk(w.pk_of(i).begin(), w.pk_of(i).end());
            const Bytes msg(w.msg(i).begin(), w.msg(i).end());
            const Bytes ctx(w.ctx(i).begin(), w.ctx(i).end());
            Bytes want;
            switch (w.iface) {
            case ppq::Interface::External:
                want = ppq::cpuref::mu_external(w.params, pk, msg, ctx);
                break;
            case ppq::Interface::Internal:
                want = ppq::cpuref::mu_internal(w.params, pk, msg);
                break;
            case ppq::Interface::Mu:
                want = msg;
                break;
            }
            CHECK(mu_of(w.iface, tr_of(pk), msg, ctx) == want);
        }
    }
}

TEST_CASE(module6_hashlib_vectors) {
    std::size_t n = 0;
    for (const auto &[id, vectors] : mldsa_test::load(TEST_VECTORS_DIR "/mldsa_hash.txt")) {
        for (const mldsa_test::Vector &v : vectors) {
            CHECK(tr_of(v.pk) == v.tr);
            // module 6 hashed pure ML-DSA with an empty context
            CHECK(mu_of(ppq::Interface::External, v.tr, v.msg, {}) == v.mu);
            ++n;
        }
    }
    CHECK(n > 0);
}

TEST_CASE(lengths_across_block_boundaries_match_naive) {
    // SHAKE256 absorbs 136 bytes per block; tr (64) + prefix (2) + ctx + msg straddles them
    ppq::Drbg rng(0x6d75, "pk");
    Bytes pk(ppq::ML_DSA_65.pk_bytes);
    rng.fill(pk.data(), pk.size());
    const Bytes tr = tr_of(pk);
    CHECK(tr == naive::tr_of(pk));

    for (const std::size_t ctx_len : {0u, 1u, 69u, 70u, 71u, 255u}) {
        for (const std::size_t msg_len : {0u, 1u, 70u, 135u, 136u, 137u, 1000u}) {
            Bytes ctx(ctx_len);
            Bytes msg(msg_len);
            ppq::Drbg(0x6d75, "ctx", ctx_len).fill(ctx.data(), ctx.size());
            ppq::Drbg(0x6d75, "msg", msg_len).fill(msg.data(), msg.size());
            CHECK(mu_of(ppq::Interface::External, tr, msg, ctx) ==
                  naive::mu_external(pk, msg, ctx));
            CHECK(mu_of(ppq::Interface::Internal, tr, msg, {}) ==
                  naive::shake256(naive::concat({tr, msg}), 64));
        }
    }
}

} // namespace

int main() { return check::run_all(); }
