#include <string>
#include <vector>

#include "check.h"
#include "mldsa_vectors.h"
#include "ppq/mldsa.h"
#include "ppq/shake_vectors.h"

namespace {

using mldsa_test::Bytes;

TEST_CASE(cpu_reference_matches_vectors) {
    const auto groups = mldsa_test::load(std::string(TEST_VECTORS_DIR) + "/mldsa_hash.txt");
    CHECK(groups.size() == std::size(ppq::PARAM_SETS));

    for (const auto &[id, vectors] : groups) {
        const ppq::ParamSet *params = ppq::find_param_set(id);
        CHECK(params != nullptr);
        if (params == nullptr) {
            continue;
        }

        const ppq::BatchLayout layout{*params, vectors.front().msg.size()};
        const std::size_t n = vectors.size();
        Bytes pk(n * layout.pk_stride());
        Bytes msg(n * layout.msg_stride());
        Bytes digest(n * layout.digest_stride());
        Bytes matrix(n * layout.matrix_stride());

        for (std::size_t s = 0; s < n; ++s) {
            mldsa_test::pack(layout, vectors[s], s, pk.data(), msg.data());
        }

        const ppq::BatchView view{pk.data(), msg.data(), digest.data(), matrix.data(), n};
        ppq::hash_mu_cpu(layout, view);
        ppq::expand_a_cpu(layout, view);

        for (std::size_t s = 0; s < n; ++s) {
            const std::uint8_t *d = digest.data() + s * layout.digest_stride();
            const std::string what = "ML-DSA-" + std::to_string(id) + " #" + std::to_string(s);
            CHECK_EQ_BYTES(mldsa_test::bytes(d, ppq::TR_BYTES), vectors[s].tr, what + " tr");
            CHECK_EQ_BYTES(mldsa_test::bytes(d + ppq::TR_BYTES, ppq::MU_BYTES), vectors[s].mu,
                           what + " mu");
            CHECK_EQ_BYTES(mldsa_test::stream_digests(matrix.data() + s * layout.matrix_stride(),
                                                      layout.matrix_entries()),
                           vectors[s].stream_digests, what + " ExpandA");
        }
    }
}

TEST_CASE(expand_a_seed_puts_column_before_row) {
    Bytes rho(ppq::SEED_BYTES);
    for (std::size_t i = 0; i < rho.size(); ++i) {
        rho[i] = static_cast<std::uint8_t>(i);
    }

    Bytes seed = rho;
    seed.push_back(1); // j
    seed.push_back(0); // i
    const Bytes want =
        ppq::shake_cpu(ppq::ShakeVariant::Shake128, seed.data(), seed.size(), ppq::EXPAND_A_BYTES);

    Bytes got(ppq::EXPAND_A_BYTES);
    ppq::expand_a_one(rho.data(), 0, 1, got.data());
    CHECK_EQ_BYTES(got, want, "A[0][1]");

    ppq::expand_a_one(rho.data(), 1, 0, got.data());
    CHECK(got != want);
}

TEST_CASE(public_key_sizes) {
    CHECK(ppq::find_param_set(44)->pk_bytes == 1312);
    CHECK(ppq::find_param_set(65)->pk_bytes == 1952);
    CHECK(ppq::find_param_set(87)->pk_bytes == 2592);
    CHECK(ppq::find_param_set(50) == nullptr);
}

} // namespace

int main() { return check::run_all(); }
