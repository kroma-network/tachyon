#include "tachyon/crypto/commitments/pedersen/pedersen.h"

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"

namespace tachyon::crypto {

namespace {

class PedersenTest : public testing::Test {
 public:
  constexpr static size_t kMaxSize = 3;

  using VCS = Pedersen<math::bn254::G1JacobianPoint, kMaxSize>;

  static void SetUpTestSuite() { math::bn254::G1Curve::Init(); }
};

}  // namespace

TEST_F(PedersenTest, CommitPedersen) {
  VCS vcs;
  ASSERT_TRUE(vcs.Setup());

  std::vector<math::bn254::Fr> v =
      base::CreateVector(kMaxSize, []() { return math::bn254::Fr::Random(); });

  math::bn254::Fr r = math::bn254::Fr::Random();
  math::bn254::G1JacobianPoint commitment;
  ASSERT_TRUE(vcs.Commit(v, r, &commitment));

  math::VariableBaseMSM<math::bn254::G1JacobianPoint> msm;
  math::bn254::G1JacobianPoint msm_result;
  ASSERT_TRUE(msm.Run(vcs.generators(), v, &msm_result));

  EXPECT_EQ(commitment, msm_result + r * vcs.h());
}

TEST_F(PedersenTest, Copyable) {
  VCS expected;
  ASSERT_TRUE(expected.Setup());

  base::VectorBuffer write_buf;
  EXPECT_TRUE(write_buf.Write(expected));

  write_buf.set_buffer_offset(0);

  VCS value;
  EXPECT_TRUE(write_buf.Read(&value));

  EXPECT_EQ(expected.h(), value.h());
  EXPECT_EQ(expected.generators(), value.generators());
}

}  // namespace tachyon::crypto
