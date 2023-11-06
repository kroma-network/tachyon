#include "absl/strings/substitute.h"
#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#if defined(TACHYON_POLYGON_ZKEVM_BACKEND)
#include "tachyon/math/elliptic_curves/bn/bn254/fq.h"
#include "tachyon/math/elliptic_curves/secp/secp256k1/fq.h"
#include "tachyon/math/elliptic_curves/secp/secp256k1/fr.h"
#endif
#include "tachyon/math/finite_fields/prime_field_conversions.h"
#include "tachyon/math/finite_fields/prime_field_gpu_debug.h"

namespace tachyon::math {

namespace {

constexpr size_t kTestNum = 1000;

using FrGpuDebug = PrimeFieldGpuDebug<bn254::FrConfig>;

template <typename PrimeFieldType>
class PrimeFieldCorrectnessTest : public testing::Test {
 public:
  using PrimeFieldGmpType = PrimeFieldGmp<typename PrimeFieldType::Config>;

  static void SetUpTestSuite() {
    PrimeFieldGmpType::Init();
    kMontgomeryRGmp =
        PrimeFieldGmpType::FromBigInt(PrimeFieldType::Config::kMontgomeryR);

    a_gmps_.reserve(kTestNum);
    b_gmps_.reserve(kTestNum);
    as_.reserve(kTestNum);
    bs_.reserve(kTestNum);

    for (size_t i = 0; i < kTestNum; ++i) {
      PrimeFieldGmpType a_gmp = PrimeFieldGmpType::Random();
      PrimeFieldGmpType b_gmp = PrimeFieldGmpType::Random();
      PrimeFieldType a = ConvertPrimeField<PrimeFieldType>(a_gmp);
      PrimeFieldType b = ConvertPrimeField<PrimeFieldType>(b_gmp);

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

  static PrimeFieldGmpType kMontgomeryRGmp;
  static std::vector<PrimeFieldGmpType> a_gmps_;
  static std::vector<PrimeFieldGmpType> b_gmps_;
  static std::vector<PrimeFieldType> as_;
  static std::vector<PrimeFieldType> bs_;
};

template <typename PrimeFieldType>
typename PrimeFieldCorrectnessTest<PrimeFieldType>::PrimeFieldGmpType
    PrimeFieldCorrectnessTest<PrimeFieldType>::kMontgomeryRGmp;
template <typename PrimeFieldType>
std::vector<
    typename PrimeFieldCorrectnessTest<PrimeFieldType>::PrimeFieldGmpType>
    PrimeFieldCorrectnessTest<PrimeFieldType>::a_gmps_;
template <typename PrimeFieldType>
std::vector<
    typename PrimeFieldCorrectnessTest<PrimeFieldType>::PrimeFieldGmpType>
    PrimeFieldCorrectnessTest<PrimeFieldType>::b_gmps_;
template <typename PrimeFieldType>
std::vector<PrimeFieldType> PrimeFieldCorrectnessTest<PrimeFieldType>::as_;
template <typename PrimeFieldType>
std::vector<PrimeFieldType> PrimeFieldCorrectnessTest<PrimeFieldType>::bs_;

}  // namespace

#if defined(TACHYON_POLYGON_ZKEVM_BACKEND)
using PrimeFieldTypes = testing::Types<bn254::Fr, bn254::Fq, secp256k1::Fq,
                                       secp256k1::Fr, FrGpuDebug>;
#else
using PrimeFieldTypes = testing::Types<bn254::Fr, FrGpuDebug>;
#endif
TYPED_TEST_SUITE(PrimeFieldCorrectnessTest, PrimeFieldTypes);

TYPED_TEST(PrimeFieldCorrectnessTest, MontgomeryForm) {
  using F = TypeParam;
  using GmpF = PrimeFieldGmp<typename F::Config>;

  for (size_t i = 0; i < kTestNum; ++i) {
    const GmpF& a_gmp = PrimeFieldCorrectnessTest<F>::a_gmps_[i];
    const F& a = PrimeFieldCorrectnessTest<F>::as_[i];

    SCOPED_TRACE(absl::Substitute("a: $0", a_gmp.ToString()));
    GmpF a_gmp_mont = a_gmp * PrimeFieldCorrectnessTest<F>::kMontgomeryRGmp;
    ASSERT_EQ(a_gmp_mont.ToBigInt(), a.value());
  }
}

TYPED_TEST(PrimeFieldCorrectnessTest, AdditiveOperators) {
  using F = TypeParam;
  using GmpF = PrimeFieldGmp<typename F::Config>;

  for (size_t i = 0; i < kTestNum; ++i) {
    GmpF a_gmp = PrimeFieldCorrectnessTest<F>::a_gmps_[i];
    const GmpF& b_gmp = PrimeFieldCorrectnessTest<F>::b_gmps_[i];
    F a = PrimeFieldCorrectnessTest<F>::as_[i];
    const F& b = PrimeFieldCorrectnessTest<F>::bs_[i];
    SCOPED_TRACE(
        absl::Substitute("a: $0, b: $1", a_gmp.ToString(), b_gmp.ToString()));

    ASSERT_EQ(ConvertPrimeField<GmpF>(a + b), a_gmp + b_gmp);
    a += b;
    a_gmp += b_gmp;
    ASSERT_EQ(ConvertPrimeField<GmpF>(a), a_gmp);

    ASSERT_EQ(ConvertPrimeField<GmpF>(a - b), a_gmp - b_gmp);
    a -= b;
    a_gmp -= b_gmp;
    ASSERT_EQ(ConvertPrimeField<GmpF>(a), a_gmp);
  }
}

TYPED_TEST(PrimeFieldCorrectnessTest, AdditiveGroupOperators) {
  using F = TypeParam;
  using GmpF = PrimeFieldGmp<typename F::Config>;

  for (size_t i = 0; i < kTestNum; ++i) {
    GmpF a_gmp = PrimeFieldCorrectnessTest<F>::a_gmps_[i];
    F a = PrimeFieldCorrectnessTest<F>::as_[i];
    SCOPED_TRACE(absl::Substitute("a: $0", a_gmp.ToString()));

    ASSERT_EQ(ConvertPrimeField<GmpF>(-a), -a_gmp);
    a.NegInPlace();
    a_gmp.NegInPlace();
    ASSERT_EQ(ConvertPrimeField<GmpF>(a), a_gmp);

    ASSERT_EQ(ConvertPrimeField<GmpF>(a.Double()), a_gmp.Double());
    a.DoubleInPlace();
    a_gmp.DoubleInPlace();
    ASSERT_EQ(ConvertPrimeField<GmpF>(a), a_gmp);
  }
}

TYPED_TEST(PrimeFieldCorrectnessTest, MultiplicativeOperators) {
  using F = TypeParam;
  using GmpF = PrimeFieldGmp<typename F::Config>;

  for (size_t i = 0; i < kTestNum; ++i) {
    GmpF a_gmp = PrimeFieldCorrectnessTest<F>::a_gmps_[i];
    const GmpF& b_gmp = PrimeFieldCorrectnessTest<F>::b_gmps_[i];
    F a = PrimeFieldCorrectnessTest<F>::as_[i];
    const F& b = PrimeFieldCorrectnessTest<F>::bs_[i];
    SCOPED_TRACE(
        absl::Substitute("a: $0, b: $1", a_gmp.ToString(), b_gmp.ToString()));

    ASSERT_EQ(ConvertPrimeField<GmpF>(a * b), a_gmp * b_gmp);
    if constexpr (!F::Config::kIsSpecialPrime &&
                  !std::is_same_v<F, FrGpuDebug>) {
      a.FastMulInPlace(b);
      a_gmp *= b_gmp;
      ASSERT_EQ(ConvertPrimeField<GmpF>(a), a_gmp);
      a.SlowMulInPlace(b);
      a_gmp *= b_gmp;
      ASSERT_EQ(ConvertPrimeField<GmpF>(a), a_gmp);
    } else {
      a *= b;
      a_gmp *= b_gmp;
      ASSERT_EQ(ConvertPrimeField<GmpF>(a), a_gmp);
    }

    ASSERT_EQ(ConvertPrimeField<GmpF>(a / b), a_gmp / b_gmp);
    a /= b;
    a_gmp /= b_gmp;
    ASSERT_EQ(ConvertPrimeField<GmpF>(a), a_gmp);
  }
}

TYPED_TEST(PrimeFieldCorrectnessTest, MultiplicativeGroupOperators) {
  using F = TypeParam;
  using GmpF = PrimeFieldGmp<typename F::Config>;

  for (size_t i = 0; i < kTestNum; ++i) {
    GmpF a_gmp = PrimeFieldCorrectnessTest<F>::a_gmps_[i];
    F a = PrimeFieldCorrectnessTest<F>::as_[i];
    SCOPED_TRACE(absl::Substitute("a: $0", a_gmp.ToString()));

    ASSERT_EQ(ConvertPrimeField<GmpF>(a.Inverse()), a_gmp.Inverse());
    a.InverseInPlace();
    a_gmp.InverseInPlace();
    ASSERT_EQ(ConvertPrimeField<GmpF>(a), a_gmp);

    ASSERT_EQ(ConvertPrimeField<GmpF>(a.Square()), a_gmp.Square());
    a.SquareInPlace();
    a_gmp.SquareInPlace();
    ASSERT_EQ(ConvertPrimeField<GmpF>(a), a_gmp);
  }
}

TYPED_TEST(PrimeFieldCorrectnessTest, SumOfProducts) {
  using F = TypeParam;
  using GmpF = PrimeFieldGmp<typename F::Config>;

  const auto& as = PrimeFieldCorrectnessTest<F>::as_;
  const auto& bs = PrimeFieldCorrectnessTest<F>::bs_;
  const auto& a_gmps = PrimeFieldCorrectnessTest<F>::a_gmps_;
  const auto& b_gmps = PrimeFieldCorrectnessTest<F>::b_gmps_;
  ASSERT_EQ(ConvertPrimeField<GmpF>(F::SumOfProducts(as, bs)),
            GmpF::SumOfProducts(a_gmps, b_gmps));
}

}  // namespace tachyon::math
