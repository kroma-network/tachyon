#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/msm/test/msm_test_set.h"

namespace tachyon::math {

namespace {

const size_t kSize = 40;

template <typename Point>
class VariableBaseMSMTest : public testing::Test {
 public:
  static void SetUpTestSuite() { Point::Curve::Init(); }

  VariableBaseMSMTest()
      : test_set_(MSMTestSet<Point>::Random(kSize, MSMMethod::kNaive)) {}
  VariableBaseMSMTest(const VariableBaseMSMTest&) = delete;
  VariableBaseMSMTest& operator=(const VariableBaseMSMTest&) = delete;
  ~VariableBaseMSMTest() override = default;

 protected:
  MSMTestSet<Point> test_set_;
};

}  // namespace

using PointTypes =
    testing::Types<bn254::G1AffinePoint, bn254::G1ProjectivePoint,
                   bn254::G1JacobianPoint, bn254::G1PointXYZZ>;
TYPED_TEST_SUITE(VariableBaseMSMTest, PointTypes);

TYPED_TEST(VariableBaseMSMTest, DoMSM) {
  using Point = TypeParam;
  using Bucket = typename VariableBaseMSM<Point>::Bucket;

  const MSMTestSet<Point>& test_set = this->test_set_;

  VariableBaseMSM<Point> msm;
  Bucket ret;
  EXPECT_TRUE(msm.Run(test_set.bases, test_set.scalars, &ret));
  EXPECT_EQ(ret, test_set.answer);
}

}  // namespace tachyon::math
