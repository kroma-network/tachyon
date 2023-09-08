#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/msm/test/msm_test_set.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/point_xyzz.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/projective_point.h"

namespace tachyon::math {

namespace {

const size_t kSize = 40;

template <typename PointTy>
class VariableBaseMSMTest : public testing::Test {
 public:
  static void SetUpTestSuite() { PointTy::Curve::Init(); }

  VariableBaseMSMTest()
      : test_set_(MSMTestSet<PointTy>::Random(kSize, MSMMethod::kNaive)) {}
  VariableBaseMSMTest(const VariableBaseMSMTest&) = delete;
  VariableBaseMSMTest& operator=(const VariableBaseMSMTest&) = delete;
  ~VariableBaseMSMTest() override = default;

 protected:
  MSMTestSet<PointTy> test_set_;
};

}  // namespace

using PointTypes =
    testing::Types<bn254::G1AffinePoint, bn254::G1ProjectivePoint,
                   bn254::G1JacobianPoint, bn254::G1PointXYZZ>;
TYPED_TEST_SUITE(VariableBaseMSMTest, PointTypes);

TYPED_TEST(VariableBaseMSMTest, DoMSM) {
  using PointTy = TypeParam;
  using Bucket = typename VariableBaseMSM<PointTy>::Bucket;

  const MSMTestSet<PointTy>& test_set = this->test_set_;

  VariableBaseMSM<PointTy> msm;
  Bucket ret;
  EXPECT_TRUE(msm.Run(test_set.bases, test_set.scalars, &ret));
  EXPECT_EQ(ret, test_set.answer);
}

}  // namespace tachyon::math
