#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"

#include "gtest/gtest.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/test_config.h"

namespace tachyon {
namespace math {

namespace {

using Config = TestSwCurveConfig::Config;

class VariableBaseMSMTest : public ::testing::Test {
 public:
  VariableBaseMSMTest() {
    Fp7::Init();
    TestSwCurveConfig::Init();

    bases_ = base::CreateVector(
        40, []() { return JacobianPoint<Config>::Random(); });
    scalars_ = base::CreateVector(40, []() { return Fp7::Random(); });

    for (size_t i = 0; i < bases_.size(); ++i) {
      answer_ += (bases_[i] * scalars_[i]);
    }
  }
  VariableBaseMSMTest(const VariableBaseMSMTest&) = delete;
  VariableBaseMSMTest& operator=(const VariableBaseMSMTest&) = delete;
  ~VariableBaseMSMTest() override = default;

 protected:
  std::vector<JacobianPoint<Config>> bases_;
  std::vector<Fp7> scalars_;
  JacobianPoint<Config> answer_;
};

}  // namespace

TEST_F(VariableBaseMSMTest, DoMSM) {
  EXPECT_EQ(VariableBaseMSM<JacobianPoint<Config>>::DoMSM(
                bases_.begin(), bases_.end(), scalars_.begin(), scalars_.end()),
            answer_);
}

TEST_F(VariableBaseMSMTest, DoMSMWindowNAF) {
  EXPECT_EQ(VariableBaseMSM<JacobianPoint<Config>>::DoMSMWindowNAF(
                bases_.begin(), bases_.end(), scalars_.begin(), scalars_.end()),
            answer_);
}

}  // namespace math
}  // namespace tachyon
