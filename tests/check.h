#ifndef TESTS_CHECK_H_
#define TESTS_CHECK_H_

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace check {

struct Case {
    const char *name;
    void (*fn)();
};

inline std::vector<Case> &registry() {
    static std::vector<Case> cases;
    return cases;
}

inline int &case_failures() {
    static int n = 0;
    return n;
}

inline bool add(const char *name, void (*fn)()) {
    registry().push_back({name, fn});
    return true;
}

inline bool expect(bool ok, const char *expr, const char *file, int line) {
    if (!ok) {
        ++case_failures();
        std::cerr << "    " << file << ":" << line << ": CHECK(" << expr << ") failed\n";
    }
    return ok;
}

inline bool expect_bytes(const std::vector<std::uint8_t> &got,
                         const std::vector<std::uint8_t> &want, const std::string &what,
                         const char *file, int line) {
    if (got == want) {
        return true;
    }

    ++case_failures();
    std::size_t at = 0;
    while (at < got.size() && at < want.size() && got[at] == want[at]) {
        ++at;
    }
    std::cerr << "    " << file << ":" << line << ": " << what << ": " << got.size() << " vs "
              << want.size() << " bytes, first difference at byte " << at << "\n";
    return false;
}

class Skip : public std::exception {
  public:
    explicit Skip(std::string why) : _why(std::move(why)) {}
    const char *what() const noexcept override { return _why.c_str(); }

  private:
    std::string _why;
};

inline int run_all() {
    int failed = 0;
    int skipped = 0;

    for (const Case &c : registry()) {
        case_failures() = 0;
        std::cout << "[ RUN  ] " << c.name << std::endl;

        try {
            c.fn();
        } catch (const Skip &s) {
            std::cout << "[ SKIP ] " << c.name << ": " << s.what() << "\n";
            ++skipped;
            continue;
        } catch (const std::exception &e) {
            std::cerr << "    exception: " << e.what() << "\n";
            ++case_failures();
        }

        if (case_failures() == 0) {
            std::cout << "[  OK  ] " << c.name << "\n";
        } else {
            std::cout << "[ FAIL ] " << c.name << " (" << case_failures() << " checks)\n";
            ++failed;
        }
    }

    std::cout << registry().size() - static_cast<std::size_t>(failed + skipped) << " passed, "
              << failed << " failed, " << skipped << " skipped\n";
    return failed == 0 ? 0 : 1;
}

} // namespace check

#define TEST_CASE(name)                                                                            \
    static void name();                                                                            \
    [[maybe_unused]] static const bool name##_registered = ::check::add(#name, name);              \
    static void name()

#define CHECK(cond) ::check::expect((cond), #cond, __FILE__, __LINE__)
#define CHECK_EQ_BYTES(got, want, what)                                                            \
    ::check::expect_bytes((got), (want), (what), __FILE__, __LINE__)

#endif
