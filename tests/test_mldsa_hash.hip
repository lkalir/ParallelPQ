#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_test.h"
#include "mldsa_vectors.h"
#include "ppq/hash_batch.h"
#include "ppq/hash_kernels.h"

namespace {

using mldsa_test::Bytes;

constexpr std::uint8_t CANARY = 0xAA;
constexpr std::size_t SUB_BATCH = 10;
constexpr std::size_t SUB_FIRST = 3;
constexpr std::size_t SUB_COUNT = 4;
constexpr std::uint32_t TEST_SEED = 99;

void run_kernels(const ppq::BatchLayout &layout, const ppq::BatchView &v) {
    ppq::launch_hash_mu(layout, v, gpu_test::small_launch(), nullptr);
    ppq::launch_expand_a(layout, v, gpu_test::small_launch(), nullptr);
}

TEST_CASE(gpu_matches_vectors) {
    const auto groups = mldsa_test::load(std::string(TEST_VECTORS_DIR) + "/mldsa_hash.txt");

    for (const auto &[id, vectors] : groups) {
        const ppq::BatchLayout layout{*ppq::find_param_set(id), vectors.front().msg.size()};
        const std::size_t n = vectors.size();

        ppq::HostBatch host(layout, n);
        for (std::size_t s = 0; s < n; ++s) {
            mldsa_test::pack(layout, vectors[s], s, host.pk(), host.msg());
        }

        ppq::DeviceBatch dev(layout, n);
        dev.upload_async(host, 0, n, nullptr);
        run_kernels(layout, dev.view(n));
        dev.download_async(host, 0, n, nullptr);
        HIP_CHECK(hipDeviceSynchronize());

        const ppq::BatchView out = host.view();
        for (std::size_t s = 0; s < n; ++s) {
            const std::uint8_t *d = out.digest + s * layout.digest_stride();
            const std::string what = "ML-DSA-" + std::to_string(id) + " #" + std::to_string(s);
            CHECK_EQ_BYTES(mldsa_test::bytes(d, ppq::TR_BYTES), vectors[s].tr, what + " tr");
            CHECK_EQ_BYTES(mldsa_test::bytes(d + ppq::TR_BYTES, ppq::MU_BYTES), vectors[s].mu,
                           what + " mu");
            CHECK_EQ_BYTES(mldsa_test::stream_digests(out.matrix + s * layout.matrix_stride(),
                                                      layout.matrix_entries()),
                           vectors[s].stream_digests, what + " ExpandA");
        }
    }
}

TEST_CASE(gpu_seed_puts_column_before_row) {
    const ppq::BatchLayout layout{ppq::ML_DSA_44, 0};
    ppq::HostBatch host(layout, 1);
    host.fill_random(TEST_SEED);

    ppq::DeviceBatch dev(layout, 1);
    dev.upload_async(host, 0, 1, nullptr);
    run_kernels(layout, dev.view(1));
    dev.download_async(host, 0, 1, nullptr);
    HIP_CHECK(hipDeviceSynchronize());

    Bytes seed(host.pk(), host.pk() + ppq::SEED_BYTES);
    seed.push_back(1); // j
    seed.push_back(0); // i
    const Bytes want =
        ppq::shake_cpu(ppq::ShakeVariant::Shake128, seed.data(), seed.size(), ppq::EXPAND_A_BYTES);

    const std::uint8_t *a01 = host.view().matrix + 1 * ppq::EXPAND_A_BYTES;
    CHECK_EQ_BYTES(mldsa_test::bytes(a01, ppq::EXPAND_A_BYTES), want, "A[0][1]");
}

TEST_CASE(sub_range_writes_only_its_signatures) {
    const ppq::BatchLayout layout{ppq::ML_DSA_65, 33};
    ppq::HostBatch host(layout, SUB_BATCH);
    host.fill_random(TEST_SEED);

    ppq::DeviceBatch dev(layout, SUB_BATCH);
    const ppq::BatchView all = dev.view(SUB_BATCH);
    HIP_CHECK(hipMemset(all.digest, CANARY, SUB_BATCH * layout.digest_stride()));
    HIP_CHECK(hipMemset(all.matrix, CANARY, SUB_BATCH * layout.matrix_stride()));

    dev.upload_async(host, 0, SUB_BATCH, nullptr);
    run_kernels(layout, ppq::slice(layout, all, SUB_FIRST, SUB_COUNT));
    dev.download_async(host, 0, SUB_BATCH, nullptr);
    HIP_CHECK(hipDeviceSynchronize());

    Bytes digest(SUB_BATCH * layout.digest_stride());
    Bytes matrix(SUB_BATCH * layout.matrix_stride());
    const ppq::BatchView ref{host.pk(), host.msg(), digest.data(), matrix.data(), SUB_BATCH};
    ppq::hash_mu_cpu(layout, ref);
    ppq::expand_a_cpu(layout, ref);

    const ppq::BatchView gpu = host.view();
    for (std::size_t s = 0; s < SUB_BATCH; ++s) {
        const bool inside = s >= SUB_FIRST && s < SUB_FIRST + SUB_COUNT;
        const auto d =
            mldsa_test::bytes(gpu.digest + s * layout.digest_stride(), layout.digest_stride());
        const auto m =
            mldsa_test::bytes(gpu.matrix + s * layout.matrix_stride(), layout.matrix_stride());
        const std::string what = "signature " + std::to_string(s);

        if (inside) {
            CHECK_EQ_BYTES(
                d,
                mldsa_test::bytes(ref.digest + s * layout.digest_stride(), layout.digest_stride()),
                what + " digest");
            CHECK_EQ_BYTES(
                m,
                mldsa_test::bytes(ref.matrix + s * layout.matrix_stride(), layout.matrix_stride()),
                what + " matrix");
        } else {
            CHECK_EQ_BYTES(d, Bytes(d.size(), CANARY), what + " digest untouched");
            CHECK_EQ_BYTES(m, Bytes(m.size(), CANARY), what + " matrix untouched");
        }
    }
}

TEST_CASE(oversized_block_is_rejected) {
    const int max_threads = ppq::hash_kernels_max_threads();
    std::cout << "    hash kernels: at most " << max_threads << " threads per block\n";

    const ppq::BatchLayout layout{ppq::ML_DSA_44, 0};
    ppq::DeviceBatch dev(layout, 1);
    const ppq::Launch too_big{1, static_cast<std::size_t>(max_threads) + 1};
    int threw = 0;

    for (const auto launch : {ppq::launch_hash_mu, ppq::launch_expand_a}) {
        try {
            launch(layout, dev.view(1), too_big, nullptr);
        } catch (const std::invalid_argument &) {
            ++threw;
        }
    }

    CHECK(threw == 2);
}

} // namespace

int main() { return gpu_test::run_all(); }
