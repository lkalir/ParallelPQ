#ifndef TESTS_TEMP_DIR_H_
#define TESTS_TEMP_DIR_H_

#include <filesystem>
#include <string>
#include <system_error>
#include <unistd.h>

namespace test_util {

/// A fresh directory under the system temp dir, removed with everything in it
class TempDir {
  public:
    explicit TempDir(const std::string &tag) {
        const auto base = std::filesystem::temp_directory_path();
        for (int attempt = 0;; ++attempt) {
            _path = base / ("ppq-" + tag + "-" + std::to_string(::getpid()) + "-" +
                            std::to_string(attempt));
            if (std::filesystem::create_directory(_path)) {
                break;
            }
        }
    }

    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(_path, ec);
    }

    TempDir(const TempDir &) = delete;
    TempDir &operator=(const TempDir &) = delete;

    std::string file(const std::string &name) const { return (_path / name).string(); }

  private:
    std::filesystem::path _path;
};

} // namespace test_util

#endif
