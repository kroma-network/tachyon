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
  math::bn254::G1AffinePoint cpu_commitment;
  ASSERT_TRUE(vcs.Commit(v, r, &cpu_commitment));

  math::VariableBaseMSM<math::bn254::G1AffinePoint> msm;
  math::bn254::G1PointXYZZ msm_result_xyzz;
  ASSERT_TRUE(msm.Run(vcs.generators(), v, &msm_result_xyzz));
  math::bn254::G1AffinePoint msm_result_affine = msm_result_xyzz.ToAffine();

  EXPECT_EQ(cpu_commitment, (msm_result_affine + r * vcs.h()).ToAffine());

#if TACHYON_CUDA
  vcs.SetupForGpu();

  math::bn254::G1AffinePoint gpu_commitment;
  ASSERT_TRUE(vcs.Commit(v, r, &gpu_commitment));

  EXPECT_EQ(gpu_commitment, cpu_commitment);
#endif
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
  std::vector<math::bn254::G1AffinePoint> cpu_batch_commitments =
      vcs.GetBatchCommitments();
  BatchCommitmentState& state = vcs.batch_commitment_state();
  EXPECT_EQ(state.batch_mode, false);
  EXPECT_EQ(state.batch_count, size_t{0});

  math::VariableBaseMSM<math::bn254::G1AffinePoint> msm;
  std::vector<math::bn254::G1PointXYZZ> cpu_msm_results_xyzz(num_vectors);
  for (size_t i = 0; i < num_vectors; ++i) {
    ASSERT_TRUE(msm.Run(vcs.generators(), v_vec[i], &cpu_msm_results_xyzz[i]));
  }
  std::vector<math::bn254::G1AffinePoint> cpu_msm_results_affine(num_vectors);
  ASSERT_TRUE(math::bn254::G1PointXYZZ::BatchNormalize(
      cpu_msm_results_xyzz, &cpu_msm_results_affine));

  EXPECT_EQ(cpu_batch_commitments, cpu_msm_results_affine);

#if TACHYON_CUDA
  vcs.SetupForGpu();

  state.batch_mode = true;
  state.batch_count = num_vectors;
  vcs.ResizeBatchCommitments();
  for (size_t i = 0; i < num_vectors; ++i) {
    ASSERT_TRUE(vcs.Commit(v_vec[i], r_vec[i], i));
  }
  std::vector<math::bn254::G1AffinePoint> gpu_batch_commitments =
      vcs.GetBatchCommitments();
  EXPECT_EQ(state.batch_mode, false);
  EXPECT_EQ(state.batch_count, size_t{0});

  EXPECT_EQ(gpu_batch_commitments, cpu_batch_commitments);

#endif
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
