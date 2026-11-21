/// Batched GPU NTT kernels against the stage vectors, the naive transforms, and the host
/// transform bit for bit

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <vector>

#include "gpu_test.h"
#include "ntt_test_data.h"
#include "ppq/buffer.h"
#include "ppq/ntt.h"
#include "ppq/ntt_kernels.h"

namespace {

using ntt_test::Case;
using ntt_test::Poly;

std::vector<std::int32_t> flatten(const std::vector<Case> &cases) {
    std::vector<std::int32_t> out;
    for (const Case &c : cases) {
        out.insert(out.end(), c.input.begin(), c.input.end());
    }
    return out;
}

/// Runs `launch_fn` over the inputs on the GPU and returns the transformed coefficients
template <class Fn>
std::vector<std::int32_t> on_gpu(const std::vector<std::int32_t> &in, std::size_t count,
                                 Fn &&launch_fn) {
    ppq::DeviceBuffer<std::int32_t> dev(in);
    launch_fn(dev.get(), count);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::int32_t> out(in.size());
    dev.download(out);
    return out;
}

/// The host transform of each polynomial, bit for bit
std::vector<std::int32_t> on_host(std::vector<std::int32_t> in, ppq::NttDirection dir) {
    for (std::size_t p = 0; p < in.size() / ppq::N; ++p) {
        if (dir == ppq::NttDirection::Forward) {
            ppq::ntt(in.data() + p * ppq::N);
        } else {
            ppq::invntt_tomont(in.data() + p * ppq::N);
        }
    }
    return in;
}

/// Mismatching polynomials against the canonical expectations
std::size_t wrong_canonical(const std::vector<std::int32_t> &got, const std::vector<Case> &cases,
                            ppq::NttDirection dir) {
    std::size_t wrong = 0;
    for (std::size_t p = 0; p < cases.size(); ++p) {
        for (unsigned c = 0; c < ppq::N; ++c) {
            const std::int32_t v = got[p * ppq::N + c];
            const std::int32_t canon =
                dir == ppq::NttDirection::Forward ? ppq::freeze(v) : ntt_test::untomont(v);
            if (canon != cases[p].expected[c]) {
                ++wrong;
                break;
            }
        }
    }
    return wrong;
}

const ppq::Launch LAUNCHES[] = {{1, 1}, {1, 64}, {3, 64}, {64, 128}};

TEST_CASE(simple_forward_ntt_matches_vectors_and_host) {
    const auto cases = ntt_test::forward_cases();
    const auto in = flatten(cases);
    const auto want = on_host(in, ppq::NttDirection::Forward);

    for (const ppq::Launch &l : LAUNCHES) {
        const auto got = on_gpu(in, cases.size(), [&](std::int32_t *d, std::size_t n) {
            ppq::launch_ntt_simple(d, n, ppq::NttDirection::Forward, l, nullptr);
        });
        CHECK(wrong_canonical(got, cases, ppq::NttDirection::Forward) == 0);
        CHECK(got == want);
    }
}

TEST_CASE(simple_inverse_ntt_matches_naive_and_host) {
    const auto cases = ntt_test::inverse_cases();
    const auto in = flatten(cases);
    const auto want = on_host(in, ppq::NttDirection::Inverse);

    for (const ppq::Launch &l : LAUNCHES) {
        const auto got = on_gpu(in, cases.size(), [&](std::int32_t *d, std::size_t n) {
            ppq::launch_ntt_simple(d, n, ppq::NttDirection::Inverse, l, nullptr);
        });
        CHECK(wrong_canonical(got, cases, ppq::NttDirection::Inverse) == 0);
        CHECK(got == want);
    }
}

TEST_CASE(only_the_requested_polynomials_change) {
    const auto cases = ntt_test::forward_cases(4);
    const auto in = flatten(cases);
    const std::size_t half = cases.size() / 2;

    const auto got = on_gpu(in, half, [&](std::int32_t *d, std::size_t n) {
        ppq::launch_ntt_simple(d, n, ppq::NttDirection::Forward, {2, 32}, nullptr);
    });
    const auto want =
        on_host(std::vector<std::int32_t>(in.begin(),
                                          in.begin() + static_cast<std::ptrdiff_t>(half * ppq::N)),
                ppq::NttDirection::Forward);
    CHECK(std::equal(want.begin(), want.end(), got.begin()));
    CHECK(std::equal(in.begin() + static_cast<std::ptrdiff_t>(half * ppq::N), in.end(),
                     got.begin() + static_cast<std::ptrdiff_t>(half * ppq::N)));

    // An empty batch is a no-op
    const auto none = on_gpu(in, 0, [&](std::int32_t *d, std::size_t n) {
        ppq::launch_ntt_simple(d, n, ppq::NttDirection::Forward, {1, 32}, nullptr);
    });
    CHECK(none == in);
}

/// Group sizes with 1, 2 and 3 polynomials per block and one or several blocks, so the
/// batch rarely divides evenly and some groups run inactive on the last stride
std::vector<std::pair<unsigned, ppq::Launch>> coop_geometries() {
    std::vector<std::pair<unsigned, ppq::Launch>> out;
    for (const unsigned g : ppq::NTT_GROUP_SIZES) {
        for (const std::size_t per_block : {1u, 2u, 3u}) {
            for (const std::size_t blocks : {1u, 5u}) {
                out.push_back({g, {blocks, g * per_block}});
            }
        }
    }
    return out;
}

TEST_CASE(cooperative_ntt_equals_the_simple_kernel_bit_for_bit) {
    const auto fwd_cases = ntt_test::forward_cases(8);
    const auto inv_cases = ntt_test::inverse_cases(13);
    const auto fwd_in = flatten(fwd_cases);
    const auto inv_in = flatten(inv_cases);
    const auto fwd_want = on_host(fwd_in, ppq::NttDirection::Forward);
    const auto inv_want = on_host(inv_in, ppq::NttDirection::Inverse);

    for (const auto &[group, l] : coop_geometries()) {
        const auto fwd = on_gpu(fwd_in, fwd_cases.size(), [&](std::int32_t *d, std::size_t n) {
            ppq::launch_ntt_coop(d, n, ppq::NttDirection::Forward, group, l, nullptr);
        });
        const auto inv = on_gpu(inv_in, inv_cases.size(), [&](std::int32_t *d, std::size_t n) {
            ppq::launch_ntt_coop(d, n, ppq::NttDirection::Inverse, group, l, nullptr);
        });
        const bool ok = fwd == fwd_want && inv == inv_want;
        if (!ok) {
            std::cerr << "    group " << group << ", " << l.blocks << " x " << l.threads << "\n";
        }
        CHECK(ok);
    }
    CHECK(wrong_canonical(on_host(fwd_in, ppq::NttDirection::Forward), fwd_cases,
                          ppq::NttDirection::Forward) == 0);
}

TEST_CASE(cooperative_ntt_handles_partial_and_empty_batches) {
    const auto cases = ntt_test::forward_cases(3);
    const auto in = flatten(cases);
    const std::size_t part = 7;
    const auto got = on_gpu(in, part, [&](std::int32_t *d, std::size_t n) {
        ppq::launch_ntt_coop(d, n, ppq::NttDirection::Forward, 32, {2, 96}, nullptr);
    });
    const auto want =
        on_host(std::vector<std::int32_t>(in.begin(),
                                          in.begin() + static_cast<std::ptrdiff_t>(part * ppq::N)),
                ppq::NttDirection::Forward);
    CHECK(std::equal(want.begin(), want.end(), got.begin()));
    CHECK(std::equal(in.begin() + static_cast<std::ptrdiff_t>(part * ppq::N), in.end(),
                     got.begin() + static_cast<std::ptrdiff_t>(part * ppq::N)));

    const auto none = on_gpu(in, 0, [&](std::int32_t *d, std::size_t n) {
        ppq::launch_ntt_coop(d, n, ppq::NttDirection::Inverse, 64, {1, 64}, nullptr);
    });
    CHECK(none == in);
}

TEST_CASE(bad_launches_are_rejected) {
    ppq::DeviceBuffer<std::int32_t> dev(ppq::N);
    bool threw = false;
    try {
        ppq::launch_ntt_simple(dev.get(), 1, ppq::NttDirection::Forward, {1, 0}, nullptr);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
    CHECK(ppq::ntt_simple_max_threads() >= 64);

    const auto rejects = [&](unsigned group, ppq::Launch l) {
        try {
            ppq::launch_ntt_coop(dev.get(), 1, ppq::NttDirection::Forward, group, l, nullptr);
        } catch (const std::invalid_argument &) {
            return true;
        }
        return false;
    };
    CHECK(rejects(48, {1, 48})); // does not divide 128
    CHECK(rejects(32, {1, 48})); // block not a multiple of the group
    CHECK(rejects(32, {1, 0}));
    CHECK(!rejects(32, {1, 64}));
    CHECK(ppq::valid_ntt_group_size(64) && !ppq::valid_ntt_group_size(8));
}

} // namespace

int main() { return gpu_test::run_all(); }
