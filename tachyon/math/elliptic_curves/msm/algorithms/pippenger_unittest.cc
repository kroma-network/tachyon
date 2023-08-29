#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/msm/test/msm_test_set.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/point_xyzz.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/projective_point.h"

namespace tachyon::math {

namespace {

const size_t kSize = 40;

template <typename PointTy>
class PippengerTest : public testing::Test {
 public:
  using ReturnTy = typename Pippenger<PointTy>::ReturnTy;

  static void SetUpTestSuite() { PointTy::Curve::Init(); }

  PippengerTest()
      : test_set_(MSMTestSet<PointTy>::Random(kSize, /*use_msm=*/false)) {}
  PippengerTest(const PippengerTest&) = delete;
  PippengerTest& operator=(const PippengerTest&) = delete;
  ~PippengerTest() override = default;

 protected:
  MSMTestSet<PointTy> test_set_;
};

}  // namespace

using PointTypes =
    testing::Types<bn254::G1AffinePoint, bn254::G1ProjectivePoint,
                   bn254::G1JacobianPoint, bn254::G1PointXYZZ>;
TYPED_TEST_SUITE(PippengerTest, PointTypes);

TYPED_TEST(PippengerTest, Run) {
  using PointTy = TypeParam;
  using ReturnTy = typename Pippenger<PointTy>::ReturnTy;

  const MSMTestSet<PointTy>& test_set = this->test_set_;

  for (int i = 0; i < 2; ++i) {
    bool use_window_naf = i == 0;
    Pippenger<PointTy> pippenger;
    SCOPED_TRACE(absl::Substitute("use_window_naf: $0", use_window_naf));
    pippenger.SetUseMSMWindowNAForTesting(use_window_naf);
    ReturnTy ret;
    EXPECT_TRUE(pippenger.Run(test_set.bases.begin(), test_set.bases.end(),
                              test_set.scalars.begin(), test_set.scalars.end(),
                              &ret));
    EXPECT_EQ(ret, test_set.answer);
  }
}

}  // namespace tachyon::math
