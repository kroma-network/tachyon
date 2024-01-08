#include "tachyon/math/elliptic_curves/msm/algorithms/pippenger/pippenger.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bls12/bls12_381/g1.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/msm/test/msm_test_set.h"

namespace tachyon::math {

namespace {

const size_t kSize = 40;

template <typename Point>
class PippengerTest : public testing::Test {
 public:
  static void SetUpTestSuite() { Point::Curve::Init(); }

  PippengerTest()
      : test_set_(MSMTestSet<Point>::Random(kSize, MSMMethod::kNaive)) {}
  PippengerTest(const PippengerTest&) = delete;
  PippengerTest& operator=(const PippengerTest&) = delete;
  ~PippengerTest() override = default;

 protected:
  MSMTestSet<Point> test_set_;
};

}  // namespace

using PointTypes =
    testing::Types<bn254::G1AffinePoint, bn254::G1ProjectivePoint,
                   bn254::G1JacobianPoint, bn254::G1PointXYZZ,
                   // See https://github.com/kroma-network/tachyon/pull/31
                   bls12_381::G1AffinePoint>;
TYPED_TEST_SUITE(PippengerTest, PointTypes);

TYPED_TEST(PippengerTest, Run) {
  using Point = TypeParam;
  using Bucket = typename Pippenger<Point>::Bucket;

  const MSMTestSet<Point>& test_set = this->test_set_;

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
    Pippenger<Point> pippenger;
    SCOPED_TRACE(absl::Substitute("use_window_naf: $0 parallel_windows: $1",
                                  test.use_window_naf, test.parallel_windows));
    pippenger.SetUseMSMWindowNAForTesting(test.use_window_naf);
    pippenger.SetParallelWindows(test.parallel_windows);
    Bucket ret;
    EXPECT_TRUE(pippenger.Run(test_set.bases.begin(), test_set.bases.end(),
                              test_set.scalars.begin(), test_set.scalars.end(),
                              &ret));
    EXPECT_EQ(ret, test_set.answer);
  }
}

}  // namespace tachyon::math
