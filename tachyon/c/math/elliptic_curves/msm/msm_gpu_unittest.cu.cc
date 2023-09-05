#include "tachyon/c/math/elliptic_curves/msm/msm_gpu.h"

#include "gtest/gtest.h"

#include "tachyon/base/bits.h"
#include "tachyon/cc/math/elliptic_curves/bn/bn254/point_traits.h"
#include "tachyon/cc/math/elliptic_curves/point_conversions.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/msm/test/msm_test_set.h"

namespace tachyon::math {

constexpr size_t kNums[] = {32, 2, 5};

class MSMGpuTest : public testing::Test {
 public:
  static void SetUpTestSuite() {
    bn254::G1AffinePoint::Curve::Init();

    size_t max_num = *std::max_element(std::begin(kNums), std::end(kNums));
    tachyon_init_msm_gpu(base::bits::Log2Ceiling(max_num));
  }

  static void TearDownTestSuite() { tachyon_release_msm_gpu(); }

  MSMGpuTest() {
    for (size_t n : kNums) {
      test_sets.push_back(
          MSMTestSet<bn254::G1AffinePoint>::Random(n, MSMMethod::kMSM));
    }
  }

 protected:
  std::vector<MSMTestSet<bn254::G1AffinePoint>> test_sets;
};

TEST_F(MSMGpuTest, MSMPoint2) {
  for (const MSMTestSet<bn254::G1AffinePoint>& t : test_sets) {
    std::unique_ptr<tachyon_bn254_g1_jacobian> ret;
    std::vector<Point2<BigInt<4>>> bases = base::CreateVector(
        t.bases.size(), [&t](size_t i) { return t.bases[i].ToMontgomery(); });
    ret.reset(tachyon_bn254_g1_point2_msm_gpu(
        reinterpret_cast<const tachyon_bn254_g1_point2*>(bases.data()),
        bases.size(),
        reinterpret_cast<const tachyon_bn254_fr*>(t.scalars.data()),
        t.scalars.size()));
    EXPECT_EQ(cc::math::ToJacobianPoint(*ret), t.answer);
  }
}

TEST_F(MSMGpuTest, MSMG1Affine) {
  for (const MSMTestSet<bn254::G1AffinePoint>& t : test_sets) {
    std::unique_ptr<tachyon_bn254_g1_jacobian> ret;
    ret.reset(tachyon_msm_g1_affine_gpu(
        reinterpret_cast<const tachyon_bn254_g1_affine*>(t.bases.data()),
        t.bases.size(),
        reinterpret_cast<const tachyon_bn254_fr*>(t.scalars.data()),
        t.scalars.size()));
    EXPECT_EQ(cc::math::ToJacobianPoint(*ret), t.answer);
  }
}

}  // namespace tachyon::math
