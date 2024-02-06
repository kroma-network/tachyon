#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/msm/test/fixed_base_msm_test_set.h"

namespace tachyon::math {

namespace {

const size_t kSize = 40;

template <typename Point>
class FixedBaseMSMTest : public testing::Test {
 public:
  static void SetUpTestSuite() { Point::Curve::Init(); }

  FixedBaseMSMTest()
      : test_set_(FixedBaseMSMTestSet<Point>::Random(
            kSize, FixedBaseMSMMethod::kNaive)) {}
  FixedBaseMSMTest(const FixedBaseMSMTest&) = delete;
  FixedBaseMSMTest& operator=(const FixedBaseMSMTest&) = delete;
  ~FixedBaseMSMTest() override = default;

 protected:
  FixedBaseMSMTestSet<Point> test_set_;
};

}  // namespace

using PointTypes =
    testing::Types<bn254::G1AffinePoint, bn254::G1ProjectivePoint,
                   bn254::G1JacobianPoint, bn254::G1PointXYZZ>;
TYPED_TEST_SUITE(FixedBaseMSMTest, PointTypes);

TYPED_TEST(FixedBaseMSMTest, DoMSM) {
  using Point = TypeParam;
  using AddResult = typename internal::AdditiveSemigroupTraits<Point>::ReturnTy;

  const FixedBaseMSMTestSet<Point>& test_set = this->test_set_;

  FixedBaseMSM<Point> msm;
  msm.Reset(test_set.scalars.size(), test_set.base);
  std::vector<AddResult> ret(test_set.scalars.size());
  EXPECT_TRUE(msm.Run(test_set.scalars, &ret));
  EXPECT_EQ(ret, test_set.answer);
}

}  // namespace tachyon::math
