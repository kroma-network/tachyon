#include "tachyon/crypto/commitments/pedersen/pedersen.h"

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"

namespace tachyon::crypto {

namespace {

class PedersenTest : public testing::Test {
 public:
  constexpr static size_t kMaxSize = 3;

  using VCS = Pedersen<math::bn254::G1AffinePoint, kMaxSize,
                       math::bn254::G1AffinePoint>;

  static void SetUpTestSuite() { math::bn254::G1Curve::Init(); }
};

}  // namespace

TEST_F(PedersenTest, CommitPedersen) {
  VCS vcs;
  ASSERT_TRUE(vcs.Setup());

  std::vector<math::bn254::Fr> v =
      base::CreateVector(kMaxSize, []() { return math::bn254::Fr::Random(); });

  math::bn254::Fr r = math::bn254::Fr::Random();
  math::bn254::G1AffinePoint commitment;
  ASSERT_TRUE(vcs.Commit(v, r, &commitment));

  math::VariableBaseMSM<math::bn254::G1AffinePoint> msm;
  math::bn254::G1PointXYZZ msm_result_tmp;
  ASSERT_TRUE(msm.Run(vcs.generators(), v, &msm_result_tmp));
  math::bn254::G1AffinePoint msm_result = msm_result_tmp.ToAffine();

  EXPECT_EQ(commitment, (msm_result + r * vcs.h()).ToAffine());
}

TEST_F(PedersenTest, BatchCommitPedersen) {
  VCS vcs;
  ASSERT_TRUE(vcs.Setup());

  size_t num_vectors = 10;

  std::vector<std::vector<math::bn254::Fr>> v_vec =
      base::CreateVector(num_vectors, []() {
        return base::CreateVector(kMaxSize,
                                  []() { return math::bn254::Fr::Random(); });
      });

  std::vector<math::bn254::Fr> r_vec = base::CreateVector(
      num_vectors, []() { return math::bn254::Fr::Random(); });

  vcs.SetBatchMode(num_vectors);
  for (size_t i = 0; i < num_vectors; ++i) {
    ASSERT_TRUE(vcs.Commit(v_vec[i], r_vec[i], i));
  }
  std::vector<math::bn254::G1AffinePoint> batch_commitments =
      vcs.GetBatchCommitments();
  EXPECT_EQ(vcs.batch_commitment_state().batch_mode, false);
  EXPECT_EQ(vcs.batch_commitment_state().batch_count, size_t{0});

  math::VariableBaseMSM<math::bn254::G1AffinePoint> msm;
  std::vector<math::bn254::G1PointXYZZ> msm_results_tmp(num_vectors);
  for (size_t i = 0; i < num_vectors; ++i) {
    ASSERT_TRUE(msm.Run(vcs.generators(), v_vec[i], &msm_results_tmp[i]));
  }
  std::vector<math::bn254::G1AffinePoint> msm_results(num_vectors);
  ASSERT_TRUE(
      math::bn254::G1PointXYZZ::BatchNormalize(msm_results_tmp, &msm_results));

  EXPECT_EQ(batch_commitments, msm_results);
}

TEST_F(PedersenTest, Copyable) {
  VCS expected;
  ASSERT_TRUE(expected.Setup());

  base::Uint8VectorBuffer write_buf;
  ASSERT_TRUE(write_buf.Grow(base::EstimateSize(expected)));
  ASSERT_TRUE(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  VCS value;
  ASSERT_TRUE(write_buf.Read(&value));
  EXPECT_EQ(expected.generators(), value.generators());
}

}  // namespace tachyon::crypto
