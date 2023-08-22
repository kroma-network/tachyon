#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"

#include "gtest/gtest.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
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
  using ReturnTy = typename VariableBaseMSM<PointTy>::ReturnTy;

  static void SetUpTestSuite() { PointTy::Curve::Init(); }

  VariableBaseMSMTest() {
    bases_ = base::CreateVector(kSize, []() { return PointTy::Random(); });
    scalars_ = base::CreateVector(kSize, []() { return bn254::Fr::Random(); });

    answer_ = ReturnTy::Zero();
    for (size_t i = 0; i < bases_.size(); ++i) {
      answer_ += bases_[i].ScalarMul(scalars_[i].ToBigInt());
    }
  }
  VariableBaseMSMTest(const VariableBaseMSMTest&) = delete;
  VariableBaseMSMTest& operator=(const VariableBaseMSMTest&) = delete;
  ~VariableBaseMSMTest() override = default;

 protected:
  std::vector<PointTy> bases_;
  std::vector<bn254::Fr> scalars_;
  ReturnTy answer_;
};

}  // namespace

using PointTypes =
    testing::Types<bn254::G1AffinePoint, bn254::G1ProjectivePoint,
                   bn254::G1JacobianPoint, bn254::G1PointXYZZ>;
TYPED_TEST_SUITE(VariableBaseMSMTest, PointTypes);

TYPED_TEST(VariableBaseMSMTest, DoMSM) {
  using PointTy = TypeParam;

  for (int i = 0; i < 2; ++i) {
    bool use_window_naf = i == 0;
    SCOPED_TRACE(absl::Substitute("use_window_naf: $0", use_window_naf));
    EXPECT_EQ(VariableBaseMSM<PointTy>::DoMSM(
                  this->bases_.begin(), this->bases_.end(),
                  this->scalars_.begin(), this->scalars_.end(), use_window_naf),
              this->answer_);
  }
}

}  // namespace tachyon::math
