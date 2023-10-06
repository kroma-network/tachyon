#include "tachyon/crypto/commitments/pedersen/pedersen.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"

namespace tachyon::crypto {

namespace {

class PedersenTest : public testing::Test {
 public:
  static void SetUpTestSuite() { math::bn254::G1Curve::Init(); }
};

}  // namespace

TEST_F(PedersenTest, CommitPedersen) {
  const size_t max_size = 3;

  PedersenParams<math::bn254::G1JacobianPoint> params =
      PedersenParams<math::bn254::G1JacobianPoint>::Random(max_size);

  std::vector<math::bn254::Fr> v =
      base::CreateVector(max_size, []() { return math::bn254::Fr::Random(); });

  math::bn254::Fr r = math::bn254::Fr::Random();
  math::bn254::G1JacobianPoint commitment;
  ASSERT_TRUE(params.Commit(v, r, &commitment));

  math::VariableBaseMSM<math::bn254::G1JacobianPoint> msm;
  math::bn254::G1JacobianPoint msm_result;
  ASSERT_TRUE(msm.Run(params.generators(), v, &msm_result));

  EXPECT_EQ(commitment, msm_result + r * params.h());
}

}  // namespace tachyon::crypto
