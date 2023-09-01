#include "tachyon/math/elliptic_curves/msm/algorithms/pippenger_adapter.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/msm/test/msm_test_set.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/point_xyzz.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/projective_point.h"

namespace tachyon::math {

namespace {

const size_t kSize = 1024;

class PippengerAdapterTest : public testing::Test {
 public:
  static void SetUpTestSuite() { bn254::G1AffinePoint::Curve::Init(); }

  PippengerAdapterTest()
      : test_set_(
            MSMTestSet<bn254::G1AffinePoint>::Random(kSize, MSMMethod::kMSM)) {}
  PippengerAdapterTest(const PippengerAdapterTest&) = delete;
  PippengerAdapterTest& operator=(const PippengerAdapterTest&) = delete;
  ~PippengerAdapterTest() override = default;

 protected:
  MSMTestSet<bn254::G1AffinePoint> test_set_;
};

}  // namespace

TEST_F(PippengerAdapterTest, RunWithStrategy) {
  const MSMTestSet<bn254::G1AffinePoint>& test_set = this->test_set_;

  for (PippengerParallelStrategy strategy :
       {PippengerParallelStrategy::kNone,
        PippengerParallelStrategy::kParallelWindow,
        PippengerParallelStrategy::kParallelTerm,
        PippengerParallelStrategy::kParallelWindowAndTerm}) {
    PippengerAdapter<bn254::G1AffinePoint> pippenger;
    SCOPED_TRACE(absl::Substitute("strategy: $0", static_cast<int>(strategy)));
    bn254::G1JacobianPoint ret;
    EXPECT_TRUE(pippenger.RunWithStrategy(
        test_set.bases.begin(), test_set.bases.end(), test_set.scalars.begin(),
        test_set.scalars.end(), strategy, &ret));
    EXPECT_EQ(ret, test_set.answer);
  }
}

}  // namespace tachyon::math
