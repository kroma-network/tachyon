#include "tachyon/math/finite_fields/prime_field_base.h"

#include "absl/hash/hash_testing.h"
#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fq.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::math {

namespace {
template <typename PrimeField>
class PrimeFieldBaseTest : public FiniteFieldTest<PrimeField> {};

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

  F n = F(2).Pow(F::Config::kTwoAdicity);
  EXPECT_EQ(F::FromMontgomery(F::Config::kTwoAdicRootOfUnity).Pow(n.ToBigInt()),
            F::One());
}

TYPED_TEST(PrimeFieldBaseTest, LargeSubgroupOfUnity) {
  using F = TypeParam;

  if constexpr (F::Config::kHasLargeSubgroupRootOfUnity) {
    F n =
        F(2).Pow(F::Config::kTwoAdicity) *
        F(F::Config::kSmallSubgroupBase).Pow(F::Config::kSmallSubgroupAdicity);
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
        ASSERT_EQ(root.Pow(n), F::One());
      }
    }
  } else {
    for (size_t i = 0; i <= F::Config::kTwoAdicity; ++i) {
      size_t n = size_t{1} << i;
      F root;
      ASSERT_TRUE(F::GetRootOfUnity(n, &root));
      ASSERT_EQ(root.Pow(n), F::One());
    }
  }
}

TYPED_TEST(PrimeFieldBaseTest, LegendreSymbol) {
  using F = TypeParam;

  F f = F::Random();
  LegendreSymbol symbol = f.Legendre();
  F expected;
  switch (symbol) {
    case LegendreSymbol::kOne:
      expected = F::One();
      break;
    case LegendreSymbol::kMinusOne:
      expected = F(F::Config::kModulus - typename F::BigIntTy(1));
      break;
    case LegendreSymbol::kZero:
      expected = F::Zero();
      break;
  }
  EXPECT_EQ(f.Pow(F::Config::kModulusMinusOneDivTwo), expected);
}

TYPED_TEST(PrimeFieldBaseTest, Hash) {
  using F = TypeParam;

  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(
      std::make_tuple(F::Random(), F::Random())));
}

TYPED_TEST(PrimeFieldBaseTest, JsonValueConverter) {
  using F = TypeParam;

  F expected_point(3);
  std::string expected_json = R"({"value":"0x3"})";

  F p;
  std::string error;
  ASSERT_TRUE(base::ParseJson(expected_json, &p, &error));
  ASSERT_TRUE(error.empty());
  EXPECT_EQ(p, expected_point);

  std::string json = base::WriteToJson(p);
  EXPECT_EQ(json, expected_json);
}

}  // namespace tachyon::math
