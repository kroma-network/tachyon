#include "tachyon/c/math/elliptic_curves/bn/bn254/msm.h"

#include "gtest/gtest.h"

#include "tachyon/base/bits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fq_prime_field_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_traits.h"
#include "tachyon/cc/math/elliptic_curves/point_conversions.h"
#include "tachyon/math/elliptic_curves/msm/test/msm_test_set.h"

namespace tachyon::math {

constexpr size_t kNums[] = {32, 2, 5};

class MSMTest : public testing::Test {
 public:
  static void SetUpTestSuite() {
    tachyon_bn254_g1_init();

    size_t max_num = *std::max_element(std::begin(kNums), std::end(kNums));
    msm_ = tachyon_bn254_g1_create_msm(base::bits::Log2Ceiling(max_num));
    for (size_t n : kNums) {
      test_sets_.push_back(
          MSMTestSet<bn254::G1AffinePoint>::Random(n, MSMMethod::kNaive));
    }
  }

  static void TearDownTestSuite() { tachyon_bn254_g1_destroy_msm(msm_); }

 protected:
  static tachyon_bn254_g1_msm_ptr msm_;
  static std::vector<MSMTestSet<bn254::G1AffinePoint>> test_sets_;
};

tachyon_bn254_g1_msm_ptr MSMTest::msm_;
std::vector<MSMTestSet<bn254::G1AffinePoint>> MSMTest::test_sets_;

TEST_F(MSMTest, MSMPoint2) {
  for (const MSMTestSet<bn254::G1AffinePoint>& t : test_sets_) {
    std::unique_ptr<tachyon_bn254_g1_jacobian> ret;
    std::vector<Point2<BigInt<4>>> bases = base::CreateVector(
        t.bases.size(), [&t](size_t i) { return t.bases[i].ToMontgomery(); });
    ret.reset(tachyon_bn254_g1_point2_msm(
        msm_, reinterpret_cast<const tachyon_bn254_g1_point2*>(bases.data()),
        reinterpret_cast<const tachyon_bn254_fr*>(t.scalars.data()),
        t.scalars.size()));
    EXPECT_EQ(cc::math::ToJacobianPoint(*ret), t.answer.ToJacobian());
  }
}

TEST_F(MSMTest, MSMG1Affine) {
  for (const MSMTestSet<bn254::G1AffinePoint>& t : test_sets_) {
    std::unique_ptr<tachyon_bn254_g1_jacobian> ret;
    ret.reset(tachyon_bn254_g1_affine_msm(
        msm_, reinterpret_cast<const tachyon_bn254_g1_affine*>(t.bases.data()),
        reinterpret_cast<const tachyon_bn254_fr*>(t.scalars.data()),
        t.scalars.size()));
    EXPECT_EQ(cc::math::ToJacobianPoint(*ret), t.answer.ToJacobian());
  }
}

}  // namespace tachyon::math
