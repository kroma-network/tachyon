#include "tachyon/crypto/commitments/kzg/kzg.h"

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"

namespace tachyon::crypto {

namespace {

constexpr size_t K = 3;
constexpr size_t N = size_t{1} << K;
constexpr size_t kMaxDegree = N - 1;

class KZGTest : public testing::Test {
 public:
  using PCS =
      KZG<math::bn254::G1AffinePoint, kMaxDegree, math::bn254::G1AffinePoint>;
  using Domain = math::UnivariateEvaluationDomain<math::bn254::Fr, kMaxDegree>;
  using Poly = math::UnivariateDensePolynomial<math::bn254::Fr, kMaxDegree>;
  using Evals = math::UnivariateEvaluations<math::bn254::Fr, kMaxDegree>;

  static void SetUpTestSuite() { math::bn254::G1Curve::Init(); }
};

}  // namespace

TEST_F(KZGTest, UnsafeSetup) {
  PCS pcs;
  ASSERT_TRUE(pcs.UnsafeSetup(N));

  EXPECT_EQ(pcs.N(), N);
  EXPECT_EQ(pcs.g1_powers_of_tau().size(), size_t{N});
  EXPECT_EQ(pcs.g1_powers_of_tau_lagrange().size(), size_t{N});
}

TEST_F(KZGTest, CommitLagrange) {
  PCS pcs;
  ASSERT_TRUE(pcs.UnsafeSetup(N));

  Poly poly = Poly::Random(N - 1);

  math::bn254::G1AffinePoint cpu_commit;
  ASSERT_TRUE(pcs.Commit(poly.coefficients().coefficients(), &cpu_commit));

  std::unique_ptr<Domain> domain = Domain::Create(N);
  Evals poly_evals = domain->FFT(poly);

  math::bn254::G1AffinePoint cpu_commit_lagrange;
  ASSERT_TRUE(
      pcs.CommitLagrange(poly_evals.evaluations(), &cpu_commit_lagrange));

  EXPECT_EQ(cpu_commit, cpu_commit_lagrange);

#if TACHYON_CUDA
  pcs.SetupForGpu();

  math::bn254::G1AffinePoint gpu_commit;
  ASSERT_TRUE(pcs.Commit(poly.coefficients().coefficients(), &gpu_commit));

  EXPECT_EQ(gpu_commit, cpu_commit);

  math::bn254::G1AffinePoint gpu_commit_lagrange;
  ASSERT_TRUE(
      pcs.CommitLagrange(poly_evals.evaluations(), &gpu_commit_lagrange));

  EXPECT_EQ(gpu_commit_lagrange, cpu_commit_lagrange);
#endif
}

TEST_F(KZGTest, BatchCommitLagrange) {
  PCS pcs;
  ASSERT_TRUE(pcs.UnsafeSetup(N));

  size_t num_polys = 10;
  std::vector<Poly> polys =
      base::CreateVector(num_polys, []() { return Poly::Random(N - 1); });

  BatchCommitmentState state(true, num_polys);
  pcs.ResizeBatchCommitments(num_polys);
  for (size_t i = 0; i < num_polys; ++i) {
    ASSERT_TRUE(pcs.Commit(polys[i].coefficients().coefficients(), state, i));
  }
  std::vector<math::bn254::G1AffinePoint> cpu_batch_commitments =
      pcs.GetBatchCommitments(state);
  EXPECT_EQ(state.batch_mode, false);
  EXPECT_EQ(state.batch_count, size_t{0});

  std::unique_ptr<Domain> domain = Domain::Create(N);
  std::vector<Evals> poly_evals =
      base::Map(polys, [&domain](Poly& poly) { return domain->FFT(poly); });

  state.batch_mode = true;
  state.batch_count = num_polys;
  pcs.ResizeBatchCommitments(num_polys);
  for (size_t i = 0; i < num_polys; ++i) {
    ASSERT_TRUE(pcs.CommitLagrange(poly_evals[i].evaluations(), state, i));
  }
  std::vector<math::bn254::G1AffinePoint> cpu_batch_commitments_lagrange =
      pcs.GetBatchCommitments(state);
  EXPECT_EQ(state.batch_mode, false);
  EXPECT_EQ(state.batch_count, size_t{0});

  EXPECT_EQ(cpu_batch_commitments, cpu_batch_commitments_lagrange);

#if TACHYON_CUDA
  pcs.SetupForGpu();

  state.batch_mode = true;
  state.batch_count = num_polys;
  pcs.ResizeBatchCommitments(num_polys);
  for (size_t i = 0; i < num_polys; ++i) {
    ASSERT_TRUE(pcs.Commit(polys[i].coefficients().coefficients(), state, i));
  }
  std::vector<math::bn254::G1AffinePoint> gpu_batch_commitments =
      pcs.GetBatchCommitments(state);
  EXPECT_EQ(state.batch_mode, false);
  EXPECT_EQ(state.batch_count, size_t{0});

  EXPECT_EQ(gpu_batch_commitments, cpu_batch_commitments);

  state.batch_mode = true;
  state.batch_count = num_polys;
  pcs.ResizeBatchCommitments(num_polys);
  for (size_t i = 0; i < num_polys; ++i) {
    ASSERT_TRUE(pcs.CommitLagrange(poly_evals[i].evaluations(), state, i));
  }
  std::vector<math::bn254::G1AffinePoint> gpu_batch_commitments_lagrange =
      pcs.GetBatchCommitments(state);
  EXPECT_EQ(state.batch_mode, false);
  EXPECT_EQ(state.batch_count, size_t{0});

  EXPECT_EQ(gpu_batch_commitments_lagrange, cpu_batch_commitments_lagrange);
#endif
}

TEST_F(KZGTest, Downsize) {
  PCS pcs;
  ASSERT_TRUE(pcs.UnsafeSetup(N));
  ASSERT_FALSE(pcs.Downsize(N));
  ASSERT_TRUE(pcs.Downsize(N / 2));
  EXPECT_EQ(pcs.N(), N / 2);
}

TEST_F(KZGTest, Copyable) {
  PCS expected;
  ASSERT_TRUE(expected.UnsafeSetup(N));

  base::Uint8VectorBuffer write_buf;
  ASSERT_TRUE(write_buf.Grow(base::EstimateSize(expected)));
  ASSERT_TRUE(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  PCS value;
  ASSERT_TRUE(write_buf.Read(&value));

  EXPECT_EQ(expected.g1_powers_of_tau(), value.g1_powers_of_tau());
  EXPECT_EQ(expected.g1_powers_of_tau_lagrange(),
            value.g1_powers_of_tau_lagrange());
}

}  // namespace tachyon::crypto
