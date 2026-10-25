#include <stdexcept>
#include <string>

#include "check.h"
#include "ppq/launch.h"

namespace {

constexpr int MAX_THREADS = 256;

bool rejects(const ppq::Launch &launch, std::string *message = nullptr) {
    try {
        ppq::require_block_size(launch, MAX_THREADS, "test_kernel");
    } catch (const std::invalid_argument &e) {
        if (message != nullptr) {
            *message = e.what();
        }
        return true;
    }
    return false;
}

TEST_CASE(accepts_every_size_up_to_the_limit) {
    CHECK(!rejects({1, 1}));
    CHECK(!rejects({1, 64}));
    CHECK(!rejects({1, MAX_THREADS}));
}

TEST_CASE(rejects_empty_and_oversized_blocks) {
    CHECK(rejects({1, 0}));
    CHECK(rejects({1, MAX_THREADS + 1}));
}

TEST_CASE(message_names_the_kernel_and_limit) {
    std::string message;
    CHECK(rejects({1, MAX_THREADS + 1}, &message));
    CHECK(message.find("test_kernel") != std::string::npos);
    CHECK(message.find(std::to_string(MAX_THREADS)) != std::string::npos);
}

} // namespace

int main() { return check::run_all(); }
