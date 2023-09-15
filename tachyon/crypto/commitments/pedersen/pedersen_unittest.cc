#include "tachyon/crypto/commitments/pedersen/pedersen.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"

namespace tachyon::math {

namespace {

class PedersenTest : public testing::Test {
 public:
  static void SetUpTestSuite() { bn254::G1JacobianPoint::Curve::Init(); }
};

}  // namespace

TEST_F(PedersenTest, CommitPedersen) {
  const size_t max_size = 3;

  PedersenParams<bn254::G1JacobianPoint> params =
      PedersenParams<bn254::G1JacobianPoint>::Random(max_size);

  std::vector<bn254::Fr> v =
      base::CreateVector(max_size, []() { return bn254::Fr::Random(); });

  bn254::Fr r = bn254::Fr::Random();
  bn254::G1JacobianPoint commitment;
  ASSERT_TRUE(params.Commit(v, r, &commitment));

  VariableBaseMSM<bn254::G1JacobianPoint> msm;
  bn254::G1JacobianPoint msm_result;
  ASSERT_TRUE(msm.Run(params.generators(), v, &msm_result));

  EXPECT_EQ(commitment, msm_result + r * params.h());
}

}  // namespace tachyon::math
