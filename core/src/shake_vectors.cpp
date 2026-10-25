#include "ppq/shake_vectors.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "ppq/keccak.h"

namespace ppq {

const char *variant_name(ShakeVariant v) {
    return v == ShakeVariant::Shake128 ? "shake128" : "shake256";
}

unsigned variant_rate(ShakeVariant v) {
    return v == ShakeVariant::Shake128 ? SHAKE128_RATE : SHAKE256_RATE;
}

std::vector<std::uint8_t> shake_cpu(ShakeVariant v, const std::uint8_t *in, std::size_t in_len,
                                    std::size_t out_len) {
    std::vector<std::uint8_t> out(out_len);

    if (v == ShakeVariant::Shake128) {
        shake<Shake128>(in, in_len, out.data(), out_len);
    } else {
        shake<Shake256>(in, in_len, out.data(), out_len);
    }

    return out;
}

namespace {

int hex_digit(char c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
    }
    return -1;
}

} // namespace

std::vector<std::uint8_t> from_hex(std::string_view hex) {
    if (hex == "-") {
        return {};
    }

    if (hex.size() % 2 != 0) {
        throw std::runtime_error("odd-length hex string");
    }

    std::vector<std::uint8_t> out(hex.size() / 2);

    for (std::size_t i = 0; i < out.size(); ++i) {
        const int hi = hex_digit(hex[2 * i]);
        const int lo = hex_digit(hex[2 * i + 1]);
        if (hi < 0 || lo < 0) {
            throw std::runtime_error("invalid hex digit");
        }
        out[i] = static_cast<std::uint8_t>(hi << 4 | lo);
    }

    return out;
}

std::string to_hex(const std::uint8_t *data, std::size_t len) {
    static constexpr char DIGITS[] = "0123456789abcdef";
    std::string s(2 * len, '0');

    for (std::size_t i = 0; i < len; ++i) {
        s[2 * i] = DIGITS[data[i] >> 4];
        s[2 * i + 1] = DIGITS[data[i] & 0xF];
    }

    return s;
}

std::vector<ShakeVector> load_shake_vectors(const std::string &path) {
    std::ifstream file(path);

    if (!file) {
        throw std::runtime_error("cannot open vector file '" + path + "'");
    }

    std::vector<ShakeVector> vectors;
    std::string line;
    std::size_t line_no = 0;

    while (std::getline(file, line)) {
        ++line_no;

        if (const auto hash = line.find('#'); hash != std::string::npos) {
            line.erase(hash);
        }

        std::istringstream is(line);
        std::string alg;
        std::string input;
        std::size_t out_len = 0;
        std::string expected;

        if (!(is >> alg)) {
            continue;
        }

        const auto where = path + ":" + std::to_string(line_no);

        if (!(is >> input >> out_len >> expected)) {
            throw std::runtime_error(where + ": expected 4 fields");
        }

        ShakeVector v;

        if (alg == "shake128") {
            v.variant = ShakeVariant::Shake128;
        } else if (alg == "shake256") {
            v.variant = ShakeVariant::Shake256;
        } else {
            throw std::runtime_error(where + ": unknown algorithm '" + alg + "'");
        }

        try {
            v.input = from_hex(input);
            v.expected = from_hex(expected);
        } catch (const std::runtime_error &e) {
            throw std::runtime_error(where + ": " + e.what());
        }

        v.out_len = out_len;

        if (v.expected.size() != out_len) {
            throw std::runtime_error(where + ": output length does not match");
        }

        vectors.push_back(std::move(v));
    }

    return vectors;
}

std::vector<CavpFile> find_cavp_files(const std::string &dir) {
    std::vector<CavpFile> files;

    for (const auto variant : {ShakeVariant::Shake128, ShakeVariant::Shake256}) {
        const std::string bits = variant == ShakeVariant::Shake128 ? "128" : "256";

        for (const char *kind : {"ShortMsg", "LongMsg", "VariableOut"}) {
            const auto path = std::filesystem::path(dir) / ("SHAKE" + bits + kind + ".rsp");
            if (std::filesystem::exists(path)) {
                files.push_back({path.string(), variant});
            }
        }
    }

    return files;
}

std::vector<ShakeVector> load_cavp_vectors(const CavpFile &cavp) {
    std::ifstream file(cavp.path);

    if (!file) {
        throw std::runtime_error("cannot open CAVP file '" + cavp.path + "'");
    }

    constexpr std::size_t NO_LEN = static_cast<std::size_t>(-1);
    std::vector<ShakeVector> vectors;
    std::string line;
    std::size_t len_bits = NO_LEN;
    std::size_t out_bits = 0;
    std::vector<std::uint8_t> msg;

    const auto value = [](const std::string &l) {
        auto v = l.substr(l.find('=') + 1);
        v.erase(0, v.find_first_not_of(" "));
        v.erase(v.find_last_not_of(" \r]") + 1);
        return v;
    };

    while (std::getline(file, line)) {
        if (line.rfind("[Outputlen", 0) == 0 || line.rfind("Outputlen", 0) == 0) {
            out_bits = std::stoul(value(line));
        } else if (line.rfind("Len", 0) == 0) {
            len_bits = std::stoul(value(line));
        } else if (line.rfind("Msg", 0) == 0) {
            msg = from_hex(value(line));
        } else if (line.rfind("Output", 0) == 0) {
            std::vector<std::uint8_t> out = from_hex(value(line));
            if (len_bits == NO_LEN) {
                len_bits = msg.size() * 8;
            }
            if (len_bits % 8 == 0 && out_bits % 8 == 0) {
                msg.resize(len_bits / 8);
                vectors.push_back({cavp.variant, msg, out.size(), std::move(out)});
            }
            len_bits = NO_LEN;
        }
    }

    return vectors;
}

} // namespace ppq
