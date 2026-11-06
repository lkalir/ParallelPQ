/// Workload data model and the .ppq file format, on hand-built workloads (no CPU reference)

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "ppq/drbg.h"
#include "ppq/workload.h"
#include "temp_dir.h"

namespace {

using Bytes = std::vector<std::uint8_t>;

constexpr std::uint64_t TEST_SEED = 0x776f726b6c6f6164; // "workload"

Bytes filler(std::uint64_t index, std::size_t n) {
    Bytes b(n);
    ppq::Drbg(TEST_SEED, "filler", index).fill(b.data(), n);
    return b;
}

/// Two keys, four signatures with varied message and context lengths (contents are not
/// real signatures; the format does not care)
ppq::Workload sample(const ppq::ParamSet &p = ppq::ML_DSA_65) {
    ppq::Workload w;
    w.params = p;
    w.seed = 1234;
    w.add_key(filler(0, p.pk_bytes));
    w.add_key(filler(1, p.pk_bytes));
    w.add_signature(0, filler(10, p.sig_bytes), filler(20, 0), {}, ppq::Verdict::Pass);
    w.add_signature(1, filler(11, p.sig_bytes), filler(21, 100), filler(31, 255),
                    ppq::Verdict::BadCommitment, ppq::Tamper::Message);
    w.add_signature(0, filler(12, p.sig_bytes), filler(22, 1), filler(32, 3), ppq::Verdict::ZNorm,
                    ppq::Tamper::ZNorm);
    w.add_signature(1, filler(13, p.sig_bytes), filler(23, 7000), {}, ppq::Verdict::BadHint,
                    ppq::Tamper::Hint);
    return w;
}

Bytes read_file(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    return Bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void write_file(const std::string &path, const Bytes &b) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char *>(b.data()), static_cast<std::streamsize>(b.size()));
}

bool load_fails(const std::string &path, const std::string &needle) {
    try {
        ppq::load_workload(path);
    } catch (const std::runtime_error &e) {
        if (std::string(e.what()).find(needle) == std::string::npos) {
            std::cerr << "    unexpected message: " << e.what() << "\n";
            return false;
        }
        return true;
    }
    return false;
}

bool invalid(const ppq::Workload &w) {
    try {
        w.validate();
    } catch (const std::runtime_error &) {
        return true;
    }
    return false;
}

TEST_CASE(accessors_follow_offsets) {
    const ppq::Workload w = sample();
    CHECK(w.key_count() == 2);
    CHECK(w.sig_count() == 4);
    CHECK(w.pass_count() == 1);
    CHECK(w.msg(0).empty());
    CHECK(w.msg(1).size() == 100);
    CHECK(w.msg(3).size() == 7000);
    CHECK(w.ctx(1).size() == 255);
    CHECK(w.ctx(3).empty());
    CHECK(Bytes(w.msg(2).begin(), w.msg(2).end()) == filler(22, 1));
    CHECK(Bytes(w.pk_of(3).begin(), w.pk_of(3).end()) == filler(1, w.params.pk_bytes));
    CHECK(Bytes(w.sig(2).begin(), w.sig(2).end()) == filler(12, w.params.sig_bytes));
    w.validate();
}

TEST_CASE(add_rejects_bad_input) {
    ppq::Workload w = sample();
    const auto &p = w.params;
    int threw = 0;
    const auto expect_throw = [&](const std::function<void()> &fn) {
        try {
            fn();
        } catch (const std::invalid_argument &) {
            ++threw;
        }
    };
    expect_throw([&] { w.add_key(Bytes(p.pk_bytes - 1)); });
    expect_throw([&] { w.add_signature(0, Bytes(p.sig_bytes + 1), {}, {}, ppq::Verdict::Pass); });
    expect_throw([&] { w.add_signature(2, Bytes(p.sig_bytes), {}, {}, ppq::Verdict::Pass); });
    expect_throw(
        [&] { w.add_signature(0, Bytes(p.sig_bytes), {}, Bytes(256), ppq::Verdict::Pass); });
    CHECK(threw == 4);
    CHECK(w.sig_count() == 4);
}

TEST_CASE(save_load_round_trip) {
    const test_util::TempDir dir("workload");
    for (const ppq::ParamSet &p : ppq::PARAM_SETS) {
        const ppq::Workload w = sample(p);
        const std::string path = dir.file(std::to_string(p.id) + ".ppq");
        ppq::save_workload(w, path);
        const ppq::Workload back = ppq::load_workload(path);
        CHECK(same_workload(w, back));
        CHECK(back.params.id == p.id);
        CHECK(back.seed == 1234);
        CHECK(!std::filesystem::exists(path + ".tmp"));
    }
}

TEST_CASE(empty_workload_round_trips) {
    const test_util::TempDir dir("workload");
    ppq::Workload w;
    w.params = ppq::ML_DSA_87;
    const std::string path = dir.file("empty.ppq");
    ppq::save_workload(w, path);
    const ppq::Workload back = ppq::load_workload(path);
    CHECK(back.sig_count() == 0);
    CHECK(back.key_count() == 0);
    CHECK(same_workload(w, back));
}

TEST_CASE(damaged_files_are_rejected) {
    const test_util::TempDir dir("workload");
    const std::string good = dir.file("good.ppq");
    const std::string bad = dir.file("bad.ppq");
    ppq::save_workload(sample(), good);
    const Bytes bytes = read_file(good);

    Bytes b = bytes;
    b[0] = 'X';
    write_file(bad, b);
    CHECK(load_fails(bad, "bad magic"));

    b = bytes;
    b[4] = 2; // parameter set 2 does not exist
    write_file(bad, b);
    CHECK(load_fails(bad, "unknown parameter set"));

    b = Bytes(bytes.begin(), bytes.end() - 1);
    write_file(bad, b);
    CHECK(load_fails(bad, "wrong size"));

    b = bytes;
    b.push_back(0);
    write_file(bad, b);
    CHECK(load_fails(bad, "wrong size"));

    b = bytes;
    b[16] = 0xFF; // signature count: implies a huge file, must fail before allocating
    b[23] = 0x7F;
    write_file(bad, b);
    CHECK(load_fails(bad, "wrong size"));

    b = bytes;
    b[bytes.size() / 2] ^= 0x01; // inside the signatures
    write_file(bad, b);
    CHECK(load_fails(bad, "checksum"));

    b = bytes;
    b.back() ^= 0x01; // the checksum itself
    write_file(bad, b);
    CHECK(load_fails(bad, "checksum"));

    CHECK(load_fails(dir.file("missing.ppq"), "cannot open"));
}

TEST_CASE(validate_catches_inconsistencies) {
    CHECK(!invalid(sample()));

    ppq::Workload w = sample();
    w.key_index[2] = 5;
    CHECK(invalid(w));

    w = sample();
    w.msg_offsets[2] = w.msg_offsets[1] - 1;
    CHECK(invalid(w));

    w = sample();
    w.msg_offsets.back() += 1;
    CHECK(invalid(w));

    w = sample();
    w.ctx_offsets.back() = w.ctx_data.size() + 1;
    CHECK(invalid(w));

    w = sample();
    w.expected[0] = static_cast<ppq::Verdict>(ppq::VERDICT_COUNT);
    CHECK(invalid(w));

    w = sample();
    w.tamper.pop_back();
    CHECK(invalid(w));

    w = sample();
    w.sigs.pop_back();
    CHECK(invalid(w));

    w = sample();
    w.params.id = 50;
    CHECK(invalid(w));
}

TEST_CASE(names_round_trip) {
    for (std::size_t i = 0; i < ppq::TAMPER_COUNT; ++i) {
        const auto t = static_cast<ppq::Tamper>(i);
        CHECK(ppq::parse_tamper(ppq::tamper_name(t)) == t);
    }
    CHECK(!ppq::parse_tamper("bogus").has_value());
    CHECK(ppq::expected_verdict(ppq::Tamper::None) == ppq::Verdict::Pass);
    CHECK(ppq::expected_verdict(ppq::Tamper::Hint) == ppq::Verdict::BadHint);
    CHECK(std::string(ppq::verdict_name(ppq::Verdict::ZNorm)) == "z-norm");
}

} // namespace

int main() { return check::run_all(); }
