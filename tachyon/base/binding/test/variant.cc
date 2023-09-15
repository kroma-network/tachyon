#include "tachyon/base/binding/test/variant.h"

namespace tachyon::base::test {

Variant::Variant(bool b) : b(b) {}

Variant::Variant(int i) : i(i) {}

Variant::Variant(int64_t i64) : i64(i64) {}

Variant::Variant(const std::string& s) : s(s) {}

Variant::Variant(const std::vector<int>& ivec) : ivec(ivec) {}

Variant::Variant(int i, const std::string& s) : i(i), s(s) {}

Variant::Variant(const Point& p) : p(p) {}

}  // namespace tachyon::base::test
