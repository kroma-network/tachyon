#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fq.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"

namespace tachyon::math {

namespace {
template <typename PrimeField>
class FiniteFieldTest : public testing::Test {
 public:
  static void SetUpTestSuite() { PrimeField::Init(); }
};

}  // namespace

using PrimeFieldTypes = testing::Types<bn254::Fq, bn254::Fr>;

TYPED_TEST_SUITE(FiniteFieldTest, PrimeFieldTypes);

TYPED_TEST(FiniteFieldTest, SquareRoot) {
  using F = TypeParam;

  static_assert(bn254::Fq::Config::kModulusModFourIsThree);
  static_assert(!bn254::Fr::Config::kModulusModFourIsThree);

  bool success = false;
  F f = F::Random();
  for (size_t i = 0; i < 100; ++i) {
    F sqrt;
    if (f.SquareRoot(&sqrt)) {
      EXPECT_EQ(sqrt.SquareInPlace(), f);
      success = true;
      break;
    } else {
      EXPECT_EQ(f.Legendre(), LegendreSymbol::kMinusOne);
    }
    f = F::Random();
  }
  EXPECT_TRUE(success);
}

}  // namespace tachyon::math
