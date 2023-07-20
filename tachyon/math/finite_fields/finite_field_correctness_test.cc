#include "absl/strings/substitute.h"
#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fq.h"

namespace tachyon {
namespace math {

namespace {

constexpr size_t kTestNum = 1000;

class PrimeFieldCorrectnessTest : public testing::Test {
 public:
  static void SetUpTestSuite() {
    bn254::FqGmp::Init();
    kMontgomeryRGmp = bn254::FqGmp::FromBigInt(bn254::Fq::kMontgomeryR);

    a_gmps_.reserve(kTestNum);
    b_gmps_.reserve(kTestNum);
    as_.reserve(kTestNum);
    bs_.reserve(kTestNum);

    for (size_t i = 0; i < kTestNum; ++i) {
      bn254::FqGmp a_gmp = bn254::FqGmp::Random();
      bn254::FqGmp b_gmp = bn254::FqGmp::Random();
      bn254::Fq a = bn254::Fq::FromBigInt(a_gmp.ToBigInt());
      bn254::Fq b = bn254::Fq::FromBigInt(b_gmp.ToBigInt());

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
  static std::vector<bn254::Fq> as_;
  static std::vector<bn254::Fq> bs_;
};

bn254::FqGmp PrimeFieldCorrectnessTest::kMontgomeryRGmp;
std::vector<bn254::FqGmp> PrimeFieldCorrectnessTest::a_gmps_;
std::vector<bn254::FqGmp> PrimeFieldCorrectnessTest::b_gmps_;
std::vector<bn254::Fq> PrimeFieldCorrectnessTest::as_;
std::vector<bn254::Fq> PrimeFieldCorrectnessTest::bs_;

}  // namespace

TEST_F(PrimeFieldCorrectnessTest, MontgomeryForm) {
  for (size_t i = 0; i < kTestNum; ++i) {
    const bn254::FqGmp& a_gmp = a_gmps_[i];
    const bn254::Fq& a = as_[i];

    SCOPED_TRACE(absl::Substitute("a: $0", a_gmp.ToString()));
    bn254::FqGmp a_gmp_mont = a_gmp * kMontgomeryRGmp;
    ASSERT_EQ(a_gmp_mont.ToBigInt(), a.value());
  }
}

TEST_F(PrimeFieldCorrectnessTest, AdditiveOperators) {
  for (size_t i = 0; i < kTestNum; ++i) {
    bn254::FqGmp a_gmp = a_gmps_[i];
    const bn254::FqGmp& b_gmp = b_gmps_[i];
    bn254::Fq a = as_[i];
    const bn254::Fq& b = bs_[i];
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

TEST_F(PrimeFieldCorrectnessTest, AdditiveGroupOperators) {
  for (size_t i = 0; i < kTestNum; ++i) {
    bn254::FqGmp a_gmp = a_gmps_[i];
    bn254::Fq a = as_[i];
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

TEST_F(PrimeFieldCorrectnessTest, MultiplicativeOperators) {
  for (size_t i = 0; i < 1000; ++i) {
    bn254::FqGmp a_gmp = a_gmps_[i];
    const bn254::FqGmp& b_gmp = b_gmps_[i];
    bn254::Fq a = as_[i];
    const bn254::Fq& b = bs_[i];
    SCOPED_TRACE(
        absl::Substitute("a: $0, b: $1", a_gmp.ToString(), b_gmp.ToString()));

    ASSERT_EQ((a * b).ToBigInt(), (a_gmp * b_gmp).ToBigInt());
    a.FastMulInPlace(b);
    a_gmp *= b_gmp;
    ASSERT_EQ(a.ToBigInt(), a_gmp.ToBigInt());
    a.SlowMulInPlace(b);
    a_gmp *= b_gmp;
    ASSERT_EQ(a.ToBigInt(), a_gmp.ToBigInt());

    ASSERT_EQ((a / b).ToBigInt(), (a_gmp / b_gmp).ToBigInt());
    a /= b;
    a_gmp /= b_gmp;
    ASSERT_EQ(a.ToBigInt(), a_gmp.ToBigInt());
  }
}

TEST_F(PrimeFieldCorrectnessTest, MultiplicativeGroupOperators) {
  for (size_t i = 0; i < kTestNum; ++i) {
    bn254::FqGmp a_gmp = a_gmps_[i];
    bn254::Fq a = as_[i];
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

TEST_F(PrimeFieldCorrectnessTest, SumOfProducts) {
  ASSERT_EQ(bn254::Fq::SumOfProducts(as_, bs_).ToBigInt(),
            bn254::FqGmp::SumOfProducts(a_gmps_, b_gmps_).ToBigInt());
}

}  // namespace math
}  // namespace tachyon
