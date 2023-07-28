#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"

#include "gtest/gtest.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/point_xyzz.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/projective_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/test/curve_config.h"

namespace tachyon::math {

namespace {

const size_t kSize = 40;

template <typename PointTy>
class VariableBaseMSMTest : public testing::Test {
 public:
  static void SetUpTestSuite() { PointTy::Curve::Init(); }

  VariableBaseMSMTest() {
    bases_ = base::CreateVector(kSize, []() { return PointTy::Random(); });
    scalars_ = base::CreateVector(kSize, []() { return GF7::Random(); });

    answer_ = std::make_unique<PointTy>();
    for (size_t i = 0; i < bases_.size(); ++i) {
      *answer_ += bases_[i].ScalarMul(scalars_[i].ToBigInt());
    }
  }
  VariableBaseMSMTest(const VariableBaseMSMTest&) = delete;
  VariableBaseMSMTest& operator=(const VariableBaseMSMTest&) = delete;
  ~VariableBaseMSMTest() override = default;

 protected:
  std::vector<PointTy> bases_;
  std::vector<GF7> scalars_;
  std::unique_ptr<PointTy> answer_;
};

}  // namespace

using PointTypes =
    testing::Types<test::ProjectivePoint, test::JacobianPoint, test::PointXYZZ>;
TYPED_TEST_SUITE(VariableBaseMSMTest, PointTypes);

TYPED_TEST(VariableBaseMSMTest, DoMSM) {
  using PointTy = TypeParam;

  for (int i = 0; i < 2; ++i) {
    bool use_window_naf = i == 0;
    SCOPED_TRACE(absl::Substitute("use_window_naf: $0", use_window_naf));
    EXPECT_EQ(VariableBaseMSM<PointTy>::DoMSM(
                  this->bases_.begin(), this->bases_.end(),
                  this->scalars_.begin(), this->scalars_.end(), use_window_naf),
              *this->answer_);
  }
}

}  // namespace tachyon::math
