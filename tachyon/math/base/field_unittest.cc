#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::math {

namespace {

class FieldTest : public FiniteFieldTest<GF7> {};

}  // namespace

TEST(FieldTest, SumOfProductsSerial) {
  std::vector<GF7> a = {GF7(2), GF7(3), GF7(4)};
  std::vector<GF7> b = {GF7(1), GF7(2), GF7(3)};

  GF7 result = Field<GF7>::SumOfProductsSerial(a, b);
  EXPECT_EQ(result, GF7(6));
}

TEST(FieldTest, SumOfProductsParallel) {
#if defined(TACHYON_HAS_OPENMP)
  size_t thread_nums = static_cast<size_t>(omp_get_max_threads());
#else
  size_t thread_nums = 1;
#endif

  std::vector<GF7> a =
      base::CreateVector(thread_nums * 10, []() { return GF7::Random(); });
  std::vector<GF7> b =
      base::CreateVector(thread_nums * 10, []() { return GF7::Random(); });

  EXPECT_EQ(Field<GF7>::SumOfProducts(a, b),
            Field<GF7>::SumOfProductsSerial(a, b));
}

}  // namespace tachyon::math
