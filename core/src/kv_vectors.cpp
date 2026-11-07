#include "ppq/kv_vectors.h"

#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

#include "ppq/shake_vectors.h"

namespace ppq {
namespace {

std::string trim(const std::string &s) {
    const auto begin = s.find_first_not_of(" \t\r");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = s.find_last_not_of(" \t\r");
    return s.substr(begin, end - begin + 1);
}

} // namespace

const std::string &KvRecord::str(const std::string &key) const {
    const auto it = _fields.find(key);
    if (it == _fields.end()) {
        throw std::runtime_error(where() + ": record has no '" + key + "'");
    }
    return it->second;
}

std::int64_t KvRecord::integer(const std::string &key) const {
    const std::string &s = str(key);
    errno = 0;
    char *end = nullptr;
    const long long v = std::strtoll(s.c_str(), &end, 10);
    if (s.empty() || errno != 0 || *end != '\0') {
        throw std::runtime_error(where() + ": '" + key + "' is not an integer");
    }
    return v;
}

std::vector<std::uint8_t> KvRecord::bytes(const std::string &key) const {
    try {
        return from_hex(str(key));
    } catch (const std::runtime_error &e) {
        throw std::runtime_error(where() + ": '" + key + "': " + e.what());
    }
}

std::vector<std::int32_t> KvRecord::ints(const std::string &key) const {
    const std::string &s = str(key);
    std::vector<std::int32_t> out;
    const char *p = s.c_str();
    while (*p != '\0') {
        while (*p == ' ') {
            ++p;
        }
        if (*p == '\0') {
            break;
        }
        errno = 0;
        char *end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p || errno != 0 || v < INT32_MIN || v > INT32_MAX ||
            (*end != ' ' && *end != '\0')) {
            throw std::runtime_error(where() + ": '" + key + "' has a bad integer");
        }
        out.push_back(static_cast<std::int32_t>(v));
        p = end;
    }
    return out;
}

void KvRecord::set(const std::string &key, std::string value) {
    if (!_fields.emplace(key, std::move(value)).second) {
        throw std::runtime_error(where() + ": '" + key + "' appears twice");
    }
}

std::vector<KvRecord> load_kv_records(const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open vector file '" + path + "'");
    }

    std::vector<KvRecord> records;
    bool open = false;
    std::string line;
    int line_no = 0;

    while (std::getline(in, line)) {
        ++line_no;
        const std::string t = trim(line);
        if (t.empty()) {
            open = false;
            continue;
        }
        if (t[0] == '#') {
            continue;
        }

        const auto eq = t.find('=');
        if (eq == std::string::npos) {
            throw std::runtime_error(path + ":" + std::to_string(line_no) +
                                     ": expected 'key = value'");
        }
        if (!open) {
            records.emplace_back(path, line_no);
            open = true;
        }
        records.back().set(trim(t.substr(0, eq)), trim(t.substr(eq + 1)));
    }
    return records;
}

} // namespace ppq
