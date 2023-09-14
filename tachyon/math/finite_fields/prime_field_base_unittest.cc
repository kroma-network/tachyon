#include "tachyon/math/finite_fields/prime_field_base.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fq.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"

namespace tachyon::math {

namespace {
template <typename PrimeFieldType>
class PrimeFieldBaseTest : public testing::Test {
 public:
  static void SetUpTestSuite() { PrimeFieldType::Init(); }
};

}  // namespace

using PrimeFieldTypes = testing::Types<bn254::Fq, bn254::Fr>;

TYPED_TEST_SUITE(PrimeFieldBaseTest, PrimeFieldTypes);

TYPED_TEST(PrimeFieldBaseTest, TwoAdicRootOfUnity) {
  using F = TypeParam;

  F n = F(2).Pow(BigInt<1>(F::Config::kTwoAdicity));
  EXPECT_EQ(F::FromMontgomery(F::Config::kTwoAdicRootOfUnity).Pow(n.ToBigInt()),
            F::One());
}

TYPED_TEST(PrimeFieldBaseTest, LargeSubgroupOfUnity) {
  using F = TypeParam;

  if constexpr (F::Config::kHasLargeSubgroupRootOfUnity) {
    F n = F(2).Pow(BigInt<1>(F::Config::kTwoAdicity)) *
          F(F::Config::kSmallSubgroupBase)
              .Pow(BigInt<1>(F::Config::kSmallSubgroupAdicity));
    EXPECT_EQ(F::FromMontgomery(F::Config::kLargeSubgroupRootOfUnity)
                  .Pow(n.ToBigInt()),
              F::One());
  } else {
    GTEST_SKIP() << "No LargeSubgroupRootOfUnity";
  }
}

TYPED_TEST(PrimeFieldBaseTest, GetRootOfUnity) {
  using F = TypeParam;

  F root;
  size_t n;
  if constexpr (F::Config::kHasLargeSubgroupRootOfUnity) {
    for (size_t i = 0; i <= F::Config::kTwoAdicity; ++i) {
      for (size_t j = 0; j <= F::Config::kSmallSubgroupAdicity; ++j) {
        n = (static_cast<size_t>(1) << i) *
            std::pow(static_cast<size_t>(F::Config::kSmallSubgroupBase), j);

        ASSERT_TRUE(F::GetRootOfUnity(n, &root));
        ASSERT_EQ(root.Pow(BigInt<1>(n)), F::One());
      }
    }
  } else {
    for (size_t i = 0; i <= F::Config::kTwoAdicity; ++i) {
      n = static_cast<size_t>(1) << i;
      ASSERT_TRUE(F::GetRootOfUnity(n, &root));
      ASSERT_EQ(root.Pow(BigInt<1>(n)), F::One());
    }
  }
}

}  // namespace tachyon::math
