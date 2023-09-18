#ifndef TACHYON_BASE_BINDING_TEST_VARIANT_H_
#define TACHYON_BASE_BINDING_TEST_VARIANT_H_

#include <stdint.h>

#include "tachyon/base/binding/test/point.h"

namespace tachyon::base::test {

struct Variant {
  explicit Variant(bool b);
  explicit Variant(int i);
  explicit Variant(int64_t i64);
  explicit Variant(const std::string& s);
  explicit Variant(const std::vector<int>& ivec);
  Variant(int i, const std::string& s);
  explicit Variant(const Point& p);

  bool b = false;
  int i = 0;
  int64_t i64 = 0;
  std::string s;
  std::vector<int> ivec;
  Point p;
};

}  // namespace tachyon::base::test

#endif  // TACHYON_BASE_BINDING_TEST_VARIANT_H_
