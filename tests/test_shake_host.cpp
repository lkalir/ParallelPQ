#include <algorithm>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "ppq/keccak.h"
#include "ppq/shake_vectors.h"

namespace {

using Bytes = std::vector<std::uint8_t>;
using ppq::ShakeVariant;

constexpr ShakeVariant VARIANTS[] = {ShakeVariant::Shake128, ShakeVariant::Shake256};
constexpr int RANDOM_SPLITS = 200;
constexpr std::size_t SPLIT_INPUT_BYTES = 1000;
constexpr std::size_t SPLIT_OUTPUT_BYTES = 900;
constexpr std::uint32_t TEST_SEED = 12345;

std::string g_vectors_dir = TEST_VECTORS_DIR;

const std::vector<ppq::ShakeVector> &vectors() {
    static const auto v = ppq::load_shake_vectors(g_vectors_dir + "/shake.txt");
    return v;
}

Bytes random_bytes(std::mt19937 &rng, std::size_t n) {
    Bytes b(n);
    for (auto &x : b) {
        x = static_cast<std::uint8_t>(rng());
    }
    return b;
}

template <class Fn> void with_sponge(ShakeVariant v, Fn &&fn) {
    if (v == ShakeVariant::Shake128) {
        fn(ppq::Shake128{});
    } else {
        fn(ppq::Shake256{});
    }
}

/// Lengths that sum to n; zero-length pieces included
std::vector<std::size_t> random_split(std::mt19937 &rng, std::size_t n) {
    std::vector<std::size_t> parts;
    std::uniform_int_distribution<std::size_t> piece(0, 300);

    while (n > 0) {
        const std::size_t p = std::min(piece(rng), n);
        parts.push_back(p);
        n -= p;
    }

    return parts;
}

TEST_CASE(vector_file_covers_both_variants) {
    std::size_t counts[2] = {0, 0};
    for (const auto &v : vectors()) {
        ++counts[static_cast<int>(v.variant)];
    }
    CHECK(counts[0] > 0);
    CHECK(counts[1] > 0);
}

TEST_CASE(known_answers) {
    for (const auto &v : vectors()) {
        const Bytes got = ppq::shake_cpu(v.variant, v.input.data(), v.input.size(), v.out_len);
        CHECK_EQ_BYTES(got, v.expected,
                       std::string(ppq::variant_name(v.variant)) + " in=" +
                           std::to_string(v.input.size()) + " out=" + std::to_string(v.out_len));
    }
}

TEST_CASE(padding_byte_shared_at_rate_minus_one) {
    for (const ShakeVariant variant : VARIANTS) {
        const std::size_t len = ppq::variant_rate(variant) - 1;
        bool found = false;

        for (const auto &v : vectors()) {
            if (v.variant == variant && v.input.size() == len) {
                found = true;
                const Bytes got = ppq::shake_cpu(variant, v.input.data(), len, v.out_len);
                CHECK_EQ_BYTES(got, v.expected, ppq::variant_name(variant));
            }
        }

        CHECK(found);
    }
}

TEST_CASE(split_absorb_matches_one_shot) {
    std::mt19937 rng(TEST_SEED);

    for (const ShakeVariant variant : VARIANTS) {
        for (int trial = 0; trial < RANDOM_SPLITS; ++trial) {
            const Bytes msg = random_bytes(rng, SPLIT_INPUT_BYTES);
            const Bytes want = ppq::shake_cpu(variant, msg.data(), msg.size(), 64);
            const auto parts = random_split(rng, msg.size());

            with_sponge(variant, [&](auto sponge) {
                const std::uint8_t *p = msg.data();
                for (const std::size_t n : parts) {
                    sponge.absorb(p, n);
                    p += n;
                }
                Bytes got(64);
                sponge.squeeze(got.data(), got.size());
                CHECK_EQ_BYTES(got, want, "split absorb");
            });
        }
    }
}

TEST_CASE(split_squeeze_matches_one_shot) {
    std::mt19937 rng(TEST_SEED + 1);

    for (const ShakeVariant variant : VARIANTS) {
        for (int trial = 0; trial < RANDOM_SPLITS; ++trial) {
            const Bytes msg = random_bytes(rng, 33);
            const Bytes want = ppq::shake_cpu(variant, msg.data(), msg.size(), SPLIT_OUTPUT_BYTES);
            const auto parts = random_split(rng, SPLIT_OUTPUT_BYTES);

            with_sponge(variant, [&](auto sponge) {
                sponge.absorb(msg.data(), msg.size());
                Bytes got(SPLIT_OUTPUT_BYTES);
                std::uint8_t *p = got.data();
                for (const std::size_t n : parts) {
                    sponge.squeeze(p, n);
                    p += n;
                }
                CHECK_EQ_BYTES(got, want, "split squeeze");
            });
        }
    }
}

TEST_CASE(shorter_output_is_a_prefix) {
    const Bytes msg = {'a', 'b', 'c'};

    for (const ShakeVariant variant : VARIANTS) {
        const Bytes long_out = ppq::shake_cpu(variant, msg.data(), msg.size(), 64);
        const Bytes short_out = ppq::shake_cpu(variant, msg.data(), msg.size(), 32);
        CHECK_EQ_BYTES(short_out, Bytes(long_out.begin(), long_out.begin() + 32), "prefix");
    }
}

TEST_CASE(variants_differ) {
    const Bytes msg = {'a', 'b', 'c'};
    const Bytes a = ppq::shake_cpu(ShakeVariant::Shake128, msg.data(), msg.size(), 32);
    const Bytes b = ppq::shake_cpu(ShakeVariant::Shake256, msg.data(), msg.size(), 32);
    CHECK(a != b);
}

TEST_CASE(absorb_after_squeeze_is_rejected) {
    const Bytes msg = {'a', 'b', 'c'};
    const Bytes want = ppq::shake_cpu(ShakeVariant::Shake128, msg.data(), msg.size(), 64);

    ppq::Shake128 sponge;
    CHECK(sponge.absorb(msg.data(), msg.size()));

    Bytes got(64);
    sponge.squeeze(got.data(), 32);
    CHECK(!sponge.absorb(msg.data(), msg.size()));
    sponge.squeeze(got.data() + 32, 32);

    CHECK_EQ_BYTES(got, want, "output after rejected absorb");
}

TEST_CASE(nist_cavp_files) {
    const std::string dir = (std::filesystem::path(g_vectors_dir).parent_path() / "cavp").string();
    const auto files = ppq::find_cavp_files(dir);

    if (files.empty()) {
        throw check::Skip("no SHAKE*.rsp files in " + dir);
    }

    for (const auto &f : files) {
        const auto entries = ppq::load_cavp_vectors(f);
        const std::string name = std::filesystem::path(f.path).filename().string();
        CHECK(!entries.empty());

        for (const auto &e : entries) {
            const Bytes got = ppq::shake_cpu(e.variant, e.input.data(), e.input.size(), e.out_len);
            CHECK_EQ_BYTES(got, e.expected, name);
        }

        std::cout << "    " << name << ": " << entries.size() << " entries\n";
    }
}

} // namespace

int main(int argc, char **argv) {
    if (argc > 1) {
        g_vectors_dir = argv[1];
    }
    return check::run_all();
}
