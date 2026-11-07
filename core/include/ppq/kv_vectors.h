#ifndef PPQ_KV_VECTORS_H_
#define PPQ_KV_VECTORS_H_

/// Records of "key = value" lines, separated by blank lines; '#' starts a comment line.
///
/// Used by the ACVP and stage vector files:
///
///   tcId = 1
///   params = 44
///   pk = 8f36...      (hex)
///   z = -12 4077 ...  (space-separated integers)

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ppq {

class KvRecord {
  public:
    KvRecord(std::string file, int line) : _file(std::move(file)), _line(line) {}

    bool has(const std::string &key) const { return _fields.count(key) != 0; }

    /// The raw value; throws std::runtime_error naming the record if the key is missing
    const std::string &str(const std::string &key) const;
    std::int64_t integer(const std::string &key) const;
    /// Hex; an empty value is zero bytes
    std::vector<std::uint8_t> bytes(const std::string &key) const;
    /// Space-separated decimal integers
    std::vector<std::int32_t> ints(const std::string &key) const;

    /// "file:line" of the record's first line, for error messages
    std::string where() const { return _file + ":" + std::to_string(_line); }

    void set(const std::string &key, std::string value);

  private:
    std::string _file;
    int _line;
    std::map<std::string, std::string> _fields;
};

/// Throws std::runtime_error on a missing file, a line without '=', or a repeated key
std::vector<KvRecord> load_kv_records(const std::string &path);

} // namespace ppq

#endif
