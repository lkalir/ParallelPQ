#include "ppq/workload.h"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

#include "ppq/keccak.h"

namespace ppq {

// The arrays are written as raw memory; every machine ParallelPQ targets is little-endian.
static_assert(std::endian::native == std::endian::little, ".ppq I/O assumes little-endian");
static_assert(sizeof(Verdict) == 1 && sizeof(Tamper) == 1);

const char *interface_name(Interface iface) {
    switch (iface) {
    case Interface::External:
        return "external";
    case Interface::Internal:
        return "internal";
    case Interface::Mu:
        return "mu";
    }
    return "unknown";
}

std::optional<Interface> parse_interface(std::string_view name) {
    for (std::size_t i = 0; i < INTERFACE_COUNT; ++i) {
        const auto iface = static_cast<Interface>(i);
        if (name == interface_name(iface)) {
            return iface;
        }
    }
    return std::nullopt;
}

std::optional<Verdict> parse_verdict(std::string_view name) {
    for (std::size_t i = 0; i < VERDICT_COUNT; ++i) {
        const auto v = static_cast<Verdict>(i);
        if (name == verdict_name(v)) {
            return v;
        }
    }
    return std::nullopt;
}

const char *verdict_name(Verdict v) {
    switch (v) {
    case Verdict::Pass:
        return "pass";
    case Verdict::BadCommitment:
        return "bad-commitment";
    case Verdict::ZNorm:
        return "z-norm";
    case Verdict::BadHint:
        return "bad-hint";
    }
    return "unknown";
}

const char *tamper_name(Tamper t) {
    switch (t) {
    case Tamper::None:
        return "none";
    case Tamper::Message:
        return "message";
    case Tamper::Commitment:
        return "commitment";
    case Tamper::ZNorm:
        return "z-norm";
    case Tamper::Hint:
        return "hint";
    }
    return "unknown";
}

std::optional<Tamper> parse_tamper(std::string_view name) {
    for (std::size_t i = 0; i < TAMPER_COUNT; ++i) {
        const auto t = static_cast<Tamper>(i);
        if (name == tamper_name(t)) {
            return t;
        }
    }
    return std::nullopt;
}

Verdict expected_verdict(Tamper t) {
    switch (t) {
    case Tamper::None:
        return Verdict::Pass;
    case Tamper::Message:
    case Tamper::Commitment:
        return Verdict::BadCommitment;
    case Tamper::ZNorm:
        return Verdict::ZNorm;
    case Tamper::Hint:
        return Verdict::BadHint;
    }
    return Verdict::BadCommitment;
}

std::size_t Workload::pass_count() const {
    return static_cast<std::size_t>(std::count(expected.begin(), expected.end(), Verdict::Pass));
}

ByteView Workload::pk(std::size_t key) const {
    return ByteView(pks).subspan(key * params.pk_bytes, params.pk_bytes);
}

ByteView Workload::sig(std::size_t i) const {
    return ByteView(sigs).subspan(i * params.sig_bytes, params.sig_bytes);
}

ByteView Workload::msg(std::size_t i) const {
    return ByteView(msg_data).subspan(msg_offsets[i], msg_offsets[i + 1] - msg_offsets[i]);
}

ByteView Workload::ctx(std::size_t i) const {
    return ByteView(ctx_data).subspan(ctx_offsets[i], ctx_offsets[i + 1] - ctx_offsets[i]);
}

std::uint32_t Workload::add_key(ByteView pk) {
    if (pk.size() != params.pk_bytes) {
        throw std::invalid_argument("public key has the wrong length for " +
                                    std::string(params.name));
    }
    const auto index = static_cast<std::uint32_t>(key_count());
    pks.insert(pks.end(), pk.begin(), pk.end());
    return index;
}

void Workload::add_signature(std::uint32_t key, ByteView sig, ByteView msg, ByteView ctx,
                             Verdict verdict, Tamper how) {
    if (sig.size() != params.sig_bytes) {
        throw std::invalid_argument("signature has the wrong length for " +
                                    std::string(params.name));
    }
    if (key >= key_count()) {
        throw std::invalid_argument("signature refers to a key that does not exist");
    }
    if (ctx.size() > MAX_CONTEXT_BYTES) {
        throw std::invalid_argument("context strings are at most 255 bytes");
    }
    if (iface != Interface::External && !ctx.empty()) {
        throw std::invalid_argument("only the external interface has context strings");
    }
    if (iface == Interface::Mu && msg.size() != MU_BYTES) {
        throw std::invalid_argument("mu workloads hold 64-byte mu values as messages");
    }

    sigs.insert(sigs.end(), sig.begin(), sig.end());
    key_index.push_back(key);
    msg_data.insert(msg_data.end(), msg.begin(), msg.end());
    msg_offsets.push_back(msg_data.size());
    ctx_data.insert(ctx_data.end(), ctx.begin(), ctx.end());
    ctx_offsets.push_back(ctx_data.size());
    expected.push_back(verdict);
    tamper.push_back(how);
}

namespace {

[[noreturn]] void invalid(const std::string &what) {
    throw std::runtime_error("invalid workload: " + what);
}

void check_offsets(const std::vector<std::uint64_t> &offsets, std::size_t count,
                   std::size_t data_bytes, const char *what, std::size_t max_entry) {
    if (offsets.size() != count + 1) {
        invalid(std::string(what) + " offsets: expected " + std::to_string(count + 1) +
                " entries, got " + std::to_string(offsets.size()));
    }
    if (offsets.front() != 0 || offsets.back() != data_bytes) {
        invalid(std::string(what) + " offsets do not span the data");
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (offsets[i + 1] < offsets[i] || offsets[i + 1] - offsets[i] > max_entry) {
            invalid(std::string(what) + " " + std::to_string(i) + " has a bad length");
        }
    }
}

} // namespace

void Workload::validate() const {
    const ParamSet *known = find_param_set(params.id);
    if (known == nullptr) {
        invalid("unknown parameter set " + std::to_string(params.id));
    }
    if (known->pk_bytes != params.pk_bytes || known->sig_bytes != params.sig_bytes) {
        invalid("parameter set sizes do not match ML-DSA-" + std::to_string(params.id));
    }

    const std::size_t n = sig_count();
    if (pks.size() % params.pk_bytes != 0) {
        invalid("public key array is not a whole number of keys");
    }
    if (sigs.size() != n * params.sig_bytes) {
        invalid("signature array does not hold " + std::to_string(n) + " signatures");
    }
    if (expected.size() != n || tamper.size() != n) {
        invalid("per-signature arrays have different lengths");
    }
    if (n > 0 && key_count() == 0) {
        invalid("signatures but no keys");
    }
    if (key_count() > std::numeric_limits<std::uint32_t>::max()) {
        invalid("more keys than a u32 index can address");
    }

    check_offsets(msg_offsets, n, msg_data.size(), "message",
                  std::numeric_limits<std::size_t>::max());
    check_offsets(ctx_offsets, n, ctx_data.size(), "context", MAX_CONTEXT_BYTES);

    if (static_cast<std::size_t>(iface) >= INTERFACE_COUNT) {
        invalid("unknown interface");
    }
    if (iface != Interface::External && !ctx_data.empty()) {
        invalid(std::string(interface_name(iface)) + " workloads have no contexts");
    }
    if (iface == Interface::Mu) {
        for (std::size_t i = 0; i < n; ++i) {
            if (msg_offsets[i + 1] - msg_offsets[i] != MU_BYTES) {
                invalid("signature " + std::to_string(i) + ": mu must be 64 bytes");
            }
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        if (key_index[i] >= key_count()) {
            invalid("signature " + std::to_string(i) + " refers to missing key " +
                    std::to_string(key_index[i]));
        }
        if (static_cast<std::size_t>(expected[i]) >= VERDICT_COUNT ||
            static_cast<std::size_t>(tamper[i]) >= TAMPER_COUNT) {
            invalid("signature " + std::to_string(i) + " has an unknown verdict or tamper code");
        }
    }
}

bool same_workload(const Workload &a, const Workload &b) {
    return a.params.id == b.params.id && a.iface == b.iface && a.seed == b.seed && a.pks == b.pks &&
           a.sigs == b.sigs && a.key_index == b.key_index && a.msg_offsets == b.msg_offsets &&
           a.msg_data == b.msg_data && a.ctx_offsets == b.ctx_offsets && a.ctx_data == b.ctx_data &&
           a.expected == b.expected && a.tamper == b.tamper;
}

namespace {

constexpr char MAGIC[4] = {'P', 'P', 'Q', '1'};
constexpr std::size_t HEADER_BYTES = 4 + 4 + 4 + 5 * 8;
constexpr std::size_t CHECKSUM_BYTES = 32;

struct Header {
    std::uint32_t param_id = 0;
    std::uint32_t iface = 0;
    std::uint64_t keys = 0;
    std::uint64_t sigs = 0;
    std::uint64_t msg_bytes = 0;
    std::uint64_t ctx_bytes = 0;
    std::uint64_t seed = 0;
};

void put_u32(std::uint8_t *p, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        p[i] = static_cast<std::uint8_t>(v >> (8 * i));
    }
}

std::uint32_t get_u32(const std::uint8_t *p) {
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
        v |= static_cast<std::uint32_t>(p[i]) << (8 * i);
    }
    return v;
}

/// Writes bytes to the file and into the running checksum
class HashingWriter {
  public:
    explicit HashingWriter(const std::string &path)
        : _out(path, std::ios::binary | std::ios::trunc) {
        if (!_out) {
            throw std::runtime_error("cannot create '" + path + "'");
        }
    }

    void write(const void *data, std::size_t n) {
        const auto *p = static_cast<const std::uint8_t *>(data);
        _hash.absorb(p, n);
        _out.write(reinterpret_cast<const char *>(p), static_cast<std::streamsize>(n));
    }

    template <class T> void write_vector(const std::vector<T> &v) {
        write(v.data(), v.size() * sizeof(T));
    }

    void finish(const std::string &path) {
        std::uint8_t checksum[CHECKSUM_BYTES];
        _hash.squeeze(checksum, sizeof(checksum));
        _out.write(reinterpret_cast<const char *>(checksum), sizeof(checksum));
        _out.close();
        if (!_out) {
            throw std::runtime_error("error writing '" + path + "'");
        }
    }

  private:
    std::ofstream _out;
    Shake256 _hash;
};

/// Reads bytes from the file and into the running checksum
class HashingReader {
  public:
    HashingReader(std::ifstream &in, const std::string &path) : _in(in), _path(path) {}

    void read(void *data, std::size_t n) {
        auto *p = static_cast<std::uint8_t *>(data);
        _in.read(reinterpret_cast<char *>(p), static_cast<std::streamsize>(n));
        if (static_cast<std::size_t>(_in.gcount()) != n) {
            throw std::runtime_error("'" + _path + "' is truncated");
        }
        _hash.absorb(p, n);
    }

    template <class T> void read_vector(std::vector<T> &v, std::size_t count) {
        v.resize(count);
        read(v.data(), count * sizeof(T));
    }

    void check_checksum() {
        std::uint8_t want[CHECKSUM_BYTES];
        std::uint8_t got[CHECKSUM_BYTES];
        _hash.squeeze(want, sizeof(want));
        _in.read(reinterpret_cast<char *>(got), sizeof(got));
        if (static_cast<std::size_t>(_in.gcount()) != sizeof(got)) {
            throw std::runtime_error("'" + _path + "' is truncated");
        }
        if (!std::equal(want, want + CHECKSUM_BYTES, got)) {
            throw std::runtime_error("'" + _path + "' is corrupt (checksum mismatch)");
        }
    }

  private:
    std::ifstream &_in;
    const std::string &_path;
    Shake256 _hash;
};

/// a * b + c, or nullopt on overflow
std::optional<std::uint64_t> mul_add(std::uint64_t a, std::uint64_t b, std::uint64_t c) {
    std::uint64_t m = 0;
    std::uint64_t s = 0;
    if (__builtin_mul_overflow(a, b, &m) || __builtin_add_overflow(m, c, &s)) {
        return std::nullopt;
    }
    return s;
}

/// Total file size the header implies, or nullopt if it overflows
std::optional<std::uint64_t> file_size_for(const Header &h, const ParamSet &p) {
    std::optional<std::uint64_t> total = HEADER_BYTES + CHECKSUM_BYTES;
    const auto add = [&](std::uint64_t count, std::uint64_t size) {
        if (total) {
            total = mul_add(count, size, *total);
        }
    };
    add(h.keys, p.pk_bytes);
    add(h.sigs, p.sig_bytes);
    add(h.sigs, sizeof(std::uint32_t));
    add(h.sigs, 2 * sizeof(std::uint64_t)); // message and context offsets...
    add(2, sizeof(std::uint64_t));          // ...each with one extra entry
    add(h.msg_bytes, 1);
    add(h.ctx_bytes, 1);
    add(h.sigs, 2); // verdict and tamper
    return total;
}

} // namespace

void save_workload(const Workload &w, const std::string &path) {
    w.validate();

    std::uint8_t header[HEADER_BYTES];
    std::copy(MAGIC, MAGIC + 4, header);
    put_u32(header + 4, static_cast<std::uint32_t>(w.params.id));
    put_u32(header + 8, static_cast<std::uint32_t>(w.iface));
    store_le64(header + 12, w.key_count());
    store_le64(header + 20, w.sig_count());
    store_le64(header + 28, w.msg_data.size());
    store_le64(header + 36, w.ctx_data.size());
    store_le64(header + 44, w.seed);

    const std::string tmp = path + ".tmp";
    {
        HashingWriter out(tmp);
        out.write(header, sizeof(header));
        out.write_vector(w.pks);
        out.write_vector(w.sigs);
        out.write_vector(w.key_index);
        out.write_vector(w.msg_offsets);
        out.write_vector(w.msg_data);
        out.write_vector(w.ctx_offsets);
        out.write_vector(w.ctx_data);
        out.write_vector(w.expected);
        out.write_vector(w.tamper);
        out.finish(tmp);
    }

    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        throw std::runtime_error("cannot move '" + tmp + "' to '" + path + "'");
    }
}

Workload load_workload(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open '" + path + "'");
    }

    std::error_code ec;
    const std::uint64_t actual_size = std::filesystem::file_size(path, ec);
    if (ec) {
        throw std::runtime_error("cannot stat '" + path + "'");
    }

    HashingReader r(in, path);
    std::uint8_t raw[HEADER_BYTES];
    r.read(raw, sizeof(raw));

    if (!std::equal(MAGIC, MAGIC + 4, raw)) {
        throw std::runtime_error("'" + path + "' is not a ParallelPQ workload (bad magic)");
    }

    Header h;
    h.param_id = get_u32(raw + 4);
    h.iface = get_u32(raw + 8);
    h.keys = load_le64(raw + 12);
    h.sigs = load_le64(raw + 20);
    h.msg_bytes = load_le64(raw + 28);
    h.ctx_bytes = load_le64(raw + 36);
    h.seed = load_le64(raw + 44);

    const ParamSet *params = find_param_set(static_cast<int>(h.param_id));
    if (params == nullptr) {
        throw std::runtime_error("'" + path + "' uses unknown parameter set " +
                                 std::to_string(h.param_id));
    }

    // Check the size before allocating anything the header asks for
    const auto expected_size = file_size_for(h, *params);
    if (!expected_size || *expected_size != actual_size) {
        throw std::runtime_error("'" + path + "' has the wrong size for its header (" +
                                 std::to_string(actual_size) + " bytes)");
    }

    Workload w;
    w.params = *params;
    w.iface = static_cast<Interface>(h.iface < INTERFACE_COUNT ? h.iface : INTERFACE_COUNT);
    w.seed = h.seed;
    r.read_vector(w.pks, h.keys * params->pk_bytes);
    r.read_vector(w.sigs, h.sigs * params->sig_bytes);
    r.read_vector(w.key_index, h.sigs);
    r.read_vector(w.msg_offsets, h.sigs + 1);
    r.read_vector(w.msg_data, h.msg_bytes);
    r.read_vector(w.ctx_offsets, h.sigs + 1);
    r.read_vector(w.ctx_data, h.ctx_bytes);
    r.read_vector(w.expected, h.sigs);
    r.read_vector(w.tamper, h.sigs);
    r.check_checksum();

    w.validate();
    return w;
}

} // namespace ppq
