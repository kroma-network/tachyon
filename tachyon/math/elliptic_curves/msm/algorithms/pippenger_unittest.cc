#include "tachyon/math/elliptic_curves/msm/algorithms/pippenger.h"

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
class PippengerTest : public testing::Test {
 public:
  using ReturnTy = typename Pippenger<PointTy>::ReturnTy;

  static void SetUpTestSuite() { PointTy::Curve::Init(); }

  PippengerTest()
      : test_set_(MSMTestSet<PointTy>::Random(kSize, MSMMethod::kNaive)) {}
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

  struct {
    bool use_window_naf;
    bool parallel_windows;
  } tests[] = {
    {false, false},
    {true, false},
#if defined(TACHYON_HAS_OPENMP)
    {false, true},
    {true, true},
#endif  // defined(TACHYON_HAS_OPENMP)
  };

  for (const auto& test : tests) {
    Pippenger<PointTy> pippenger;
    SCOPED_TRACE(absl::Substitute("use_window_naf: $0 parallel_windows: $1",
                                  test.use_window_naf, test.parallel_windows));
    pippenger.SetUseMSMWindowNAForTesting(test.use_window_naf);
    pippenger.SetParallelWindows(test.parallel_windows);
    ReturnTy ret;
    EXPECT_TRUE(pippenger.Run(test_set.bases.begin(), test_set.bases.end(),
                              test_set.scalars.begin(), test_set.scalars.end(),
                              &ret));
    EXPECT_EQ(ret, test_set.answer);
  }
}

}  // namespace tachyon::math
