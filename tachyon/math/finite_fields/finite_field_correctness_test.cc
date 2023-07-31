#include "absl/strings/substitute.h"
#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fq.h"
#include "tachyon/math/finite_fields/prime_field_cuda_debug.h"

namespace tachyon::math {

namespace {

constexpr size_t kTestNum = 1000;

using FqCudaDebug = PrimeFieldCudaDebug<bn254::FqConfig>;

template <typename PrimeFieldType>
class PrimeFieldCorrectnessTest : public testing::Test {
 public:
  static void SetUpTestSuite() {
    bn254::FqGmp::Init();
    kMontgomeryRGmp = bn254::FqGmp::FromBigInt(PrimeFieldType::kMontgomeryR);

    a_gmps_.reserve(kTestNum);
    b_gmps_.reserve(kTestNum);
    as_.reserve(kTestNum);
    bs_.reserve(kTestNum);

    for (size_t i = 0; i < kTestNum; ++i) {
      bn254::FqGmp a_gmp = bn254::FqGmp::Random();
      bn254::FqGmp b_gmp = bn254::FqGmp::Random();
      PrimeFieldType a = PrimeFieldType::FromMontgomery(a_gmp.ToMontgomery());
      PrimeFieldType b = PrimeFieldType::FromMontgomery(b_gmp.ToMontgomery());

      a_gmps_.push_back(std::move(a_gmp));
      b_gmps_.push_back(std::move(b_gmp));
      as_.push_back(std::move(a));
      bs_.push_back(std::move(b));
    }
  }

  static void TearDownTestSuite() {
    a_gmps_.clear();
    b_gmps_.clear();
    as_.clear();
    bs_.clear();
  }

  static bn254::FqGmp kMontgomeryRGmp;
  static std::vector<bn254::FqGmp> a_gmps_;
  static std::vector<bn254::FqGmp> b_gmps_;
  static std::vector<PrimeFieldType> as_;
  static std::vector<PrimeFieldType> bs_;
};

template <typename PrimeFieldType>
bn254::FqGmp PrimeFieldCorrectnessTest<PrimeFieldType>::kMontgomeryRGmp;
template <typename PrimeFieldType>
std::vector<bn254::FqGmp> PrimeFieldCorrectnessTest<PrimeFieldType>::a_gmps_;
template <typename PrimeFieldType>
std::vector<bn254::FqGmp> PrimeFieldCorrectnessTest<PrimeFieldType>::b_gmps_;
template <typename PrimeFieldType>
std::vector<PrimeFieldType> PrimeFieldCorrectnessTest<PrimeFieldType>::as_;
template <typename PrimeFieldType>
std::vector<PrimeFieldType> PrimeFieldCorrectnessTest<PrimeFieldType>::bs_;

}  // namespace

using PrimeFiledTypes = testing::Types<bn254::Fq, FqCudaDebug>;
TYPED_TEST_SUITE(PrimeFieldCorrectnessTest, PrimeFiledTypes);

TYPED_TEST(PrimeFieldCorrectnessTest, MontgomeryForm) {
  using F = TypeParam;

  for (size_t i = 0; i < kTestNum; ++i) {
    const bn254::FqGmp& a_gmp = PrimeFieldCorrectnessTest<F>::a_gmps_[i];
    const F& a = PrimeFieldCorrectnessTest<F>::as_[i];

    SCOPED_TRACE(absl::Substitute("a: $0", a_gmp.ToString()));
    bn254::FqGmp a_gmp_mont =
        a_gmp * PrimeFieldCorrectnessTest<F>::kMontgomeryRGmp;
    ASSERT_EQ(a_gmp_mont.ToBigInt(), a.value());
  }
}

TYPED_TEST(PrimeFieldCorrectnessTest, AdditiveOperators) {
  using F = TypeParam;

  for (size_t i = 0; i < kTestNum; ++i) {
    bn254::FqGmp a_gmp = PrimeFieldCorrectnessTest<F>::a_gmps_[i];
    const bn254::FqGmp& b_gmp = PrimeFieldCorrectnessTest<F>::b_gmps_[i];
    F a = PrimeFieldCorrectnessTest<F>::as_[i];
    const F& b = PrimeFieldCorrectnessTest<F>::bs_[i];
    SCOPED_TRACE(
        absl::Substitute("a: $0, b: $1", a_gmp.ToString(), b_gmp.ToString()));

    ASSERT_EQ((a + b).ToBigInt(), (a_gmp + b_gmp).ToBigInt());
    a += b;
    a_gmp += b_gmp;
    ASSERT_EQ(a.ToBigInt(), a_gmp.ToBigInt());

    ASSERT_EQ((a - b).ToBigInt(), (a_gmp - b_gmp).ToBigInt());
    a -= b;
    a_gmp -= b_gmp;
    ASSERT_EQ(a.ToBigInt(), a_gmp.ToBigInt());
  }
}

TYPED_TEST(PrimeFieldCorrectnessTest, AdditiveGroupOperators) {
  using F = TypeParam;

  for (size_t i = 0; i < kTestNum; ++i) {
    bn254::FqGmp a_gmp = PrimeFieldCorrectnessTest<F>::a_gmps_[i];
    F a = PrimeFieldCorrectnessTest<F>::as_[i];
    SCOPED_TRACE(absl::Substitute("a: $0", a_gmp.ToString()));

    ASSERT_EQ(a.Negative().ToBigInt(), a_gmp.Negative().ToBigInt());
    a.NegInPlace();
    a_gmp.NegInPlace();
    ASSERT_EQ(a.ToBigInt(), a_gmp.ToBigInt());

    ASSERT_EQ(a.Double().ToBigInt(), a_gmp.Double().ToBigInt());
    a.DoubleInPlace();
    a_gmp.DoubleInPlace();
    ASSERT_EQ(a.ToBigInt(), a_gmp.ToBigInt());
  }
}

TYPED_TEST(PrimeFieldCorrectnessTest, MultiplicativeOperators) {
  using F = TypeParam;

  for (size_t i = 0; i < kTestNum; ++i) {
    bn254::FqGmp a_gmp = PrimeFieldCorrectnessTest<F>::a_gmps_[i];
    const bn254::FqGmp& b_gmp = PrimeFieldCorrectnessTest<F>::b_gmps_[i];
    F a = PrimeFieldCorrectnessTest<F>::as_[i];
    const F& b = PrimeFieldCorrectnessTest<F>::bs_[i];
    SCOPED_TRACE(
        absl::Substitute("a: $0, b: $1", a_gmp.ToString(), b_gmp.ToString()));

    ASSERT_EQ((a * b).ToBigInt(), (a_gmp * b_gmp).ToBigInt());
    if constexpr (std::is_same_v<F, bn254::Fq>) {
      a.FastMulInPlace(b);
      a_gmp *= b_gmp;
      ASSERT_EQ(a.ToBigInt(), a_gmp.ToBigInt());
      a.SlowMulInPlace(b);
      a_gmp *= b_gmp;
      ASSERT_EQ(a.ToBigInt(), a_gmp.ToBigInt());
    } else {
      a *= b;
      a_gmp *= b_gmp;
      ASSERT_EQ(a.ToBigInt(), a_gmp.ToBigInt());
    }

    ASSERT_EQ((a / b).ToBigInt(), (a_gmp / b_gmp).ToBigInt());
    a /= b;
    a_gmp /= b_gmp;
    ASSERT_EQ(a.ToBigInt(), a_gmp.ToBigInt());
  }
}

TYPED_TEST(PrimeFieldCorrectnessTest, MultiplicativeGroupOperators) {
  using F = TypeParam;

  for (size_t i = 0; i < kTestNum; ++i) {
    bn254::FqGmp a_gmp = PrimeFieldCorrectnessTest<F>::a_gmps_[i];
    F a = PrimeFieldCorrectnessTest<F>::as_[i];
    SCOPED_TRACE(absl::Substitute("a: $0", a_gmp.ToString()));

    ASSERT_EQ(a.Inverse().ToBigInt(), a_gmp.Inverse().ToBigInt());
    a.InverseInPlace();
    a_gmp.InverseInPlace();
    ASSERT_EQ(a.ToBigInt(), a_gmp.ToBigInt());

    ASSERT_EQ(a.Square().ToBigInt(), a_gmp.Square().ToBigInt());
    a.SquareInPlace();
    a_gmp.SquareInPlace();
    ASSERT_EQ(a.ToBigInt(), a_gmp.ToBigInt());
  }
}

TYPED_TEST(PrimeFieldCorrectnessTest, SumOfProducts) {
  using F = TypeParam;

  const auto& as = PrimeFieldCorrectnessTest<F>::as_;
  const auto& bs = PrimeFieldCorrectnessTest<F>::bs_;
  const auto& a_gmps = PrimeFieldCorrectnessTest<F>::a_gmps_;
  const auto& b_gmps = PrimeFieldCorrectnessTest<F>::b_gmps_;
  ASSERT_EQ(F::SumOfProducts(as, bs).ToBigInt(),
            bn254::FqGmp::SumOfProducts(a_gmps, b_gmps).ToBigInt());
}

}  // namespace tachyon::math
