#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"

#include "gtest/gtest.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/test/curve_config.h"

namespace tachyon {
namespace math {

namespace {

const size_t kSize = 40;

class VariableBaseMSMTest : public testing::Test {
 public:
  static void SetUpTestSuite() { test::JacobianPoint::Curve::Init(); }

  VariableBaseMSMTest() {
    bases_ = base::CreateVector(kSize,
                                []() { return test::JacobianPoint::Random(); });
    scalars_ = base::CreateVector(kSize, []() { return GF7::Random(); });

    answer_ = std::make_unique<test::JacobianPoint>();
    for (size_t i = 0; i < bases_.size(); ++i) {
      *answer_ += bases_[i].ScalarMul(scalars_[i].ToBigInt());
    }
  }
  VariableBaseMSMTest(const VariableBaseMSMTest&) = delete;
  VariableBaseMSMTest& operator=(const VariableBaseMSMTest&) = delete;
  ~VariableBaseMSMTest() override = default;

 protected:
  std::vector<test::JacobianPoint> bases_;
  std::vector<GF7> scalars_;
  std::unique_ptr<test::JacobianPoint> answer_;
};

}  // namespace

TEST_F(VariableBaseMSMTest, DoMSM) {
  for (int i = 0; i < 2; ++i) {
    bool use_window_naf = i == 0;
    SCOPED_TRACE(absl::Substitute("use_window_naf: $0", use_window_naf));
    EXPECT_EQ(VariableBaseMSM<test::JacobianPoint>::DoMSM(
                  bases_.begin(), bases_.end(), scalars_.begin(),
                  scalars_.end(), use_window_naf),
              *answer_);
  }
}

}  // namespace math
}  // namespace tachyon
