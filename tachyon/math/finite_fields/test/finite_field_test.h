#ifndef TACHYON_MATH_FINITE_FIELDS_TEST_FINITE_FIELD_TEST_H_
#define TACHYON_MATH_FINITE_FIELDS_TEST_FINITE_FIELD_TEST_H_

#include "gtest/gtest.h"

namespace tachyon::math {

template <typename F>
class FiniteFieldTest : public testing::Test {
 public:
  static void SetUpTestSuite() { F::Init(); }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_TEST_FINITE_FIELD_TEST_H_
