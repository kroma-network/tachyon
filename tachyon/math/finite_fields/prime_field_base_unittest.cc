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

TYPED_TEST(PrimeFieldBaseTest, Decompose) {
  using F = TypeParam;

  if constexpr (F::Config::kHasLargeSubgroupRootOfUnity) {
    for (size_t i = 0; i <= F::Config::kTwoAdicity; ++i) {
      for (size_t j = 0; j <= F::Config::kSmallSubgroupAdicity; ++j) {
        size_t n = (size_t{1} << i) *
                   std::pow(size_t{F::Config::kSmallSubgroupBase}, j);

        PrimeFieldFactors unused;
        EXPECT_TRUE(F::Decompose(n, &unused));
      }
    }
  } else {
    GTEST_SKIP() << "No LargeSubgroupRootOfUnity";
  }
}

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

  if constexpr (F::Config::kHasLargeSubgroupRootOfUnity) {
    for (size_t i = 0; i <= F::Config::kTwoAdicity; ++i) {
      for (size_t j = 0; j <= F::Config::kSmallSubgroupAdicity; ++j) {
        size_t n = (size_t{1} << i) *
                   std::pow(size_t{F::Config::kSmallSubgroupBase}, j);
        F root;
        ASSERT_TRUE(F::GetRootOfUnity(n, &root));
        ASSERT_EQ(root.Pow(BigInt<1>(n)), F::One());
      }
    }
  } else {
    for (size_t i = 0; i <= F::Config::kTwoAdicity; ++i) {
      size_t n = size_t{1} << i;
      F root;
      ASSERT_TRUE(F::GetRootOfUnity(n, &root));
      ASSERT_EQ(root.Pow(BigInt<1>(n)), F::One());
    }
  }
}

TYPED_TEST(PrimeFieldBaseTest, LegendreSymbol) {
  using F = TypeParam;

  F f = F::Random();
  LegendreSymbol symbol = f.Legendre();
  EXPECT_EQ(F(static_cast<int>(symbol)),
            f.Pow(F::Config::kModulusMinusOneDivTwo));
}

}  // namespace tachyon::math
