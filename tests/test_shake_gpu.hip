#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_test.h"
#include "ppq/shake_gpu.h"
#include "ppq/shake_vectors.h"

namespace {

using Bytes = std::vector<std::uint8_t>;
using ppq::ShakeVariant;

constexpr std::size_t RANDOM_INPUTS = 10000;
constexpr std::size_t MAX_RANDOM_INPUT = 3000;
constexpr std::size_t MAX_RANDOM_OUTPUT = 400;
constexpr std::size_t MAX_MISALIGN = 7;
constexpr std::uint32_t TEST_SEED = 4242;

struct Corpus {
    std::vector<Bytes> inputs;
    std::vector<ppq::ShakeRequest> requests;
    std::vector<Bytes> expected;
};

Corpus random_corpus(std::size_t count) {
    std::mt19937 rng(TEST_SEED);
    std::uniform_int_distribution<std::size_t> in_len(0, MAX_RANDOM_INPUT);
    std::uniform_int_distribution<std::size_t> out_len(1, MAX_RANDOM_OUTPUT);

    Corpus c;
    c.inputs.resize(count);

    for (std::size_t i = 0; i < count; ++i) {
        c.inputs[i].resize(in_len(rng));
        for (auto &b : c.inputs[i]) {
            b = static_cast<std::uint8_t>(rng());
        }
        const auto variant = (rng() & 1) != 0 ? ShakeVariant::Shake128 : ShakeVariant::Shake256;
        c.requests.push_back({variant, c.inputs[i].data(), c.inputs[i].size(), out_len(rng)});
        c.expected.push_back(ppq::shake_cpu(variant, c.inputs[i].data(), c.inputs[i].size(),
                                            c.requests.back().out_len));
    }

    return c;
}

void check_all(const std::vector<Bytes> &got, const std::vector<Bytes> &want,
               const std::string &what) {
    CHECK(got.size() == want.size());
    std::size_t wrong = 0;

    for (std::size_t i = 0; i < got.size() && i < want.size(); ++i) {
        if (got[i] != want[i] && ++wrong <= 3) {
            CHECK_EQ_BYTES(got[i], want[i], what + " #" + std::to_string(i));
        }
    }

    CHECK(wrong == 0);
}

TEST_CASE(known_answers_on_gpu) {
    const auto vectors = ppq::load_shake_vectors(std::string(TEST_VECTORS_DIR) + "/shake.txt");
    std::vector<ppq::ShakeRequest> requests;
    std::vector<Bytes> expected;

    for (const auto &v : vectors) {
        requests.push_back({v.variant, v.input.data(), v.input.size(), v.out_len});
        expected.push_back(v.expected);
    }

    check_all(ppq::shake_gpu(requests, gpu_test::small_launch()), expected, "vector");
}

TEST_CASE(random_inputs_match_cpu) {
    const Corpus c = random_corpus(RANDOM_INPUTS);
    check_all(ppq::shake_gpu(c.requests, {64, 256}), c.expected, "random");
}

TEST_CASE(block_sizes_and_short_grid) {
    const Corpus c = random_corpus(RANDOM_INPUTS / 10);
    const auto warp = static_cast<std::size_t>(gpu_test::warp_size());
    const auto max_threads = static_cast<std::size_t>(ppq::shake_gpu_max_threads());

    for (const std::size_t threads :
         {std::size_t{32}, std::size_t{64}, std::size_t{256}, std::size_t{1024}}) {
        if (threads % warp != 0 || threads > max_threads) {
            continue;
        }
        check_all(ppq::shake_gpu(c.requests, {8, threads}), c.expected,
                  std::to_string(threads) + " threads");
    }

    check_all(ppq::shake_gpu(c.requests, {1, warp}), c.expected, "single block");

    const std::size_t largest = max_threads / warp * warp;
    check_all(ppq::shake_gpu(c.requests, {8, largest}), c.expected,
              "kernel limit of " + std::to_string(largest) + " threads");
}

TEST_CASE(oversized_block_is_rejected) {
    const int max_threads = ppq::shake_gpu_max_threads();
    std::cout << "    shake_jobs_kernel: at most " << max_threads << " threads per block\n";

    const std::uint8_t byte = 0;
    const std::vector<ppq::ShakeRequest> one = {{ShakeVariant::Shake128, &byte, 1, 32}};
    const std::size_t too_many = static_cast<std::size_t>(max_threads) + 1;
    bool threw = false;

    try {
        ppq::shake_gpu(one, {1, too_many});
    } catch (const std::invalid_argument &) {
        threw = true;
    }

    CHECK(threw);
}

TEST_CASE(misaligned_inputs) {
    const Corpus c = random_corpus(RANDOM_INPUTS / 10);

    for (std::size_t misalign = 1; misalign <= MAX_MISALIGN; ++misalign) {
        check_all(ppq::shake_gpu(c.requests, gpu_test::small_launch(), misalign), c.expected,
                  "misalign " + std::to_string(misalign));
    }
}

TEST_CASE(nist_cavp_files_on_gpu) {
    const std::string dir =
        (std::filesystem::path(TEST_VECTORS_DIR).parent_path() / "cavp").string();
    const auto files = ppq::find_cavp_files(dir);

    if (files.empty()) {
        throw check::Skip("no SHAKE*.rsp files in " + dir);
    }

    for (const auto &f : files) {
        const auto entries = ppq::load_cavp_vectors(f);
        const std::string name = std::filesystem::path(f.path).filename().string();
        CHECK(!entries.empty());

        std::vector<ppq::ShakeRequest> requests;
        std::vector<Bytes> expected;
        for (const auto &e : entries) {
            requests.push_back({e.variant, e.input.data(), e.input.size(), e.out_len});
            expected.push_back(e.expected);
        }

        check_all(ppq::shake_gpu(requests, {64, 128}), expected, name);
        std::cout << "    " << name << ": " << entries.size() << " entries\n";
    }
}

} // namespace

int main() { return gpu_test::run_all(); }
