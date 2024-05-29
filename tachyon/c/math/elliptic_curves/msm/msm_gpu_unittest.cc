#include "tachyon/c/math/elliptic_curves/bn/bn254/msm_gpu.h"

#include "gtest/gtest.h"

#include "tachyon/base/bits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fq_type_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_type_traits.h"
#include "tachyon/c/math/elliptic_curves/msm/algorithm.h"
#include "tachyon/math/elliptic_curves/msm/test/variable_base_msm_test_set.h"

namespace tachyon::math {

constexpr size_t kNums[] = {32, 2, 5};

class MSMGpuTest : public testing::TestWithParam<int> {
 public:
  static void SetUpTestSuite() {
    tachyon_bn254_g1_init();

    for (size_t n : kNums) {
      test_sets_.push_back(VariableBaseMSMTestSet<bn254::G1AffinePoint>::Random(
          n, VariableBaseMSMMethod::kNaive));
    }
  }

 protected:
  static std::vector<VariableBaseMSMTestSet<bn254::G1AffinePoint>> test_sets_;
};

std::vector<VariableBaseMSMTestSet<bn254::G1AffinePoint>>
    MSMGpuTest::test_sets_;

INSTANTIATE_TEST_SUITE_P(BellmanMSM, MSMGpuTest,
                         testing::Values(TACHYON_MSM_ALGO_BELLMAN_MSM));
INSTANTIATE_TEST_SUITE_P(CUZK, MSMGpuTest,
                         testing::Values(TACHYON_MSM_ALGO_CUZK));
INSTANTIATE_TEST_SUITE_P(IcicleMSM, MSMGpuTest,
                         testing::Values(TACHYON_MSM_ALGO_ICICLE_MSM));

TEST_P(MSMGpuTest, MSMPoint2) {
  size_t max_num = *std::max_element(std::begin(kNums), std::end(kNums));
  tachyon_bn254_g1_msm_gpu_ptr msm = tachyon_bn254_g1_create_msm_gpu(
      base::bits::Log2Ceiling(max_num), GetParam());

  for (const VariableBaseMSMTestSet<bn254::G1AffinePoint>& t :
       this->test_sets_) {
    std::unique_ptr<tachyon_bn254_g1_jacobian> ret;
    std::vector<Point2<bn254::Fq>> bases =
        base::CreateVector(t.bases.size(), [&t](size_t i) {
          return Point2<bn254::Fq>(t.bases[i].x(), t.bases[i].y());
        });
    ret.reset(tachyon_bn254_g1_point2_msm_gpu(
        msm, c::base::c_cast(bases.data()), c::base::c_cast(t.scalars.data()),
        t.scalars.size()));
    EXPECT_EQ(c::base::native_cast(*ret), t.answer.ToJacobian());
  }
  tachyon_bn254_g1_destroy_msm_gpu(msm);
}

TEST_P(MSMGpuTest, MSMG1Affine) {
  size_t max_num = *std::max_element(std::begin(kNums), std::end(kNums));
  tachyon_bn254_g1_msm_gpu_ptr msm = tachyon_bn254_g1_create_msm_gpu(
      base::bits::Log2Ceiling(max_num), GetParam());

  for (const VariableBaseMSMTestSet<bn254::G1AffinePoint>& t :
       this->test_sets_) {
    std::unique_ptr<tachyon_bn254_g1_jacobian> ret;
    ret.reset(tachyon_bn254_g1_affine_msm_gpu(
        msm, c::base::c_cast(t.bases.data()), c::base::c_cast(t.scalars.data()),
        t.scalars.size()));
    EXPECT_EQ(c::base::native_cast(*ret), t.answer.ToJacobian());
  }
  tachyon_bn254_g1_destroy_msm_gpu(msm);
}

}  // namespace tachyon::math
