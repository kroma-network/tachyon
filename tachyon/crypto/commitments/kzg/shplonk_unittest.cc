#include "tachyon/crypto/commitments/kzg/shplonk.h"

#include <memory>

#include "gtest/gtest.h"

#include "tachyon/crypto/transcripts/simple_transcript.h"
#include "tachyon/math/elliptic_curves/bn/bn254/bn254.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g2.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"

namespace tachyon::crypto {

namespace {

class SHPlonkTest : public testing::Test {
 public:
  constexpr static size_t K = 3;
  constexpr static size_t N = size_t{1} << K;
  constexpr static size_t kMaxDegree = N - 1;

  using PCS =
      SHPlonk<math::bn254::BN254Curve, kMaxDegree, math::bn254::G1AffinePoint,
              SimpleTranscriptReader<math::bn254::G1AffinePoint>,
              SimpleTranscriptWriter<math::bn254::G1AffinePoint>>;
  using F = PCS::Field;
  using Poly = PCS::Poly;
  using Commitment = PCS::Commitment;
  using Point = Poly::Point;
  using TranscriptReader = PCS::TranscriptReader;
  using TranscriptWriter = PCS::TranscriptWriter;
  using PolyRef = base::DeepRef<const Poly>;
  using PointRef = base::DeepRef<const Point>;
  using CommitmentRef = base::DeepRef<const Commitment>;

  static void SetUpTestSuite() {
    math::bn254::BN254Curve::Init();
    math::bn254::G1Curve::Init();
    math::bn254::G2Curve::Init();
  }

  void SetUp() override {
    KZG<math::bn254::G1AffinePoint, kMaxDegree, math::bn254::G1AffinePoint> kzg;
    pcs_ = PCS(std::move(kzg));
    ASSERT_TRUE(pcs_.UnsafeSetup(N));

    polys_ = base::CreateVector(5, []() { return Poly::Random(kMaxDegree); });
    points_ = {F(1), F(2), F(3), F(4), F(5)};

    commitments_.reserve(5);
    for (const Poly& poly : polys_) {
      Commitment commitment;
      CHECK(pcs_.Commit(poly, &commitment));
      commitments_.emplace_back(std::move(commitment));
    }

    // clang-format off
    prover_openings_.reserve(13);
    // {P₀, [x₀, x₁, x₂]}
    prover_openings_.emplace_back(PolyRef(&polys_[0]), PointRef(&points_[0]), polys_[0].Evaluate(points_[0]));
    prover_openings_.emplace_back(PolyRef(&polys_[0]), PointRef(&points_[1]), polys_[0].Evaluate(points_[1]));
    prover_openings_.emplace_back(PolyRef(&polys_[0]), PointRef(&points_[2]), polys_[0].Evaluate(points_[2]));
    // (P₁, [x₀, x₁, x₂]}
    prover_openings_.emplace_back(PolyRef(&polys_[1]), PointRef(&points_[0]), polys_[1].Evaluate(points_[0]));
    prover_openings_.emplace_back(PolyRef(&polys_[1]), PointRef(&points_[1]), polys_[1].Evaluate(points_[1]));
    prover_openings_.emplace_back(PolyRef(&polys_[1]), PointRef(&points_[2]), polys_[1].Evaluate(points_[2]));
    // (P₂, [x₀, x₁, x₂]}
    prover_openings_.emplace_back(PolyRef(&polys_[2]), PointRef(&points_[0]), polys_[2].Evaluate(points_[0]));
    prover_openings_.emplace_back(PolyRef(&polys_[2]), PointRef(&points_[1]), polys_[2].Evaluate(points_[1]));
    prover_openings_.emplace_back(PolyRef(&polys_[2]), PointRef(&points_[2]), polys_[2].Evaluate(points_[2]));
    // (P₃, [x₂, x₃]}
    prover_openings_.emplace_back(PolyRef(&polys_[3]), PointRef(&points_[2]), polys_[3].Evaluate(points_[2]));
    prover_openings_.emplace_back(PolyRef(&polys_[3]), PointRef(&points_[3]), polys_[3].Evaluate(points_[3]));
    // (P₄, [x₃, x₄]}
    prover_openings_.emplace_back(PolyRef(&polys_[4]), PointRef(&points_[3]), polys_[4].Evaluate(points_[3]));
    prover_openings_.emplace_back(PolyRef(&polys_[4]), PointRef(&points_[4]), polys_[4].Evaluate(points_[4]));

    verifier_openings_.reserve(13);
    // {C₀, [x₀, x₁, x₂]}
    verifier_openings_.emplace_back(CommitmentRef(&commitments_[0]), PointRef(&points_[0]), polys_[0].Evaluate(points_[0]));
    verifier_openings_.emplace_back(CommitmentRef(&commitments_[0]), PointRef(&points_[1]), polys_[0].Evaluate(points_[1]));
    verifier_openings_.emplace_back(CommitmentRef(&commitments_[0]), PointRef(&points_[2]), polys_[0].Evaluate(points_[2]));
    // (C₁, [x₀, x₁, x₂]}
    verifier_openings_.emplace_back(CommitmentRef(&commitments_[1]), PointRef(&points_[0]), polys_[1].Evaluate(points_[0]));
    verifier_openings_.emplace_back(CommitmentRef(&commitments_[1]), PointRef(&points_[1]), polys_[1].Evaluate(points_[1]));
    verifier_openings_.emplace_back(CommitmentRef(&commitments_[1]), PointRef(&points_[2]), polys_[1].Evaluate(points_[2]));
    // (C₂, [x₀, x₁, x₂]}
    verifier_openings_.emplace_back(CommitmentRef(&commitments_[2]), PointRef(&points_[0]), polys_[2].Evaluate(points_[0]));
    verifier_openings_.emplace_back(CommitmentRef(&commitments_[2]), PointRef(&points_[1]), polys_[2].Evaluate(points_[1]));
    verifier_openings_.emplace_back(CommitmentRef(&commitments_[2]), PointRef(&points_[2]), polys_[2].Evaluate(points_[2]));
    // (C₃, [x₂, x₃]}
    verifier_openings_.emplace_back(CommitmentRef(&commitments_[3]), PointRef(&points_[2]), polys_[3].Evaluate(points_[2]));
    verifier_openings_.emplace_back(CommitmentRef(&commitments_[3]), PointRef(&points_[3]), polys_[3].Evaluate(points_[3]));
    // (C₄, [x₃, x₄]}
    verifier_openings_.emplace_back(CommitmentRef(&commitments_[4]), PointRef(&points_[3]), polys_[4].Evaluate(points_[3]));
    verifier_openings_.emplace_back(CommitmentRef(&commitments_[4]), PointRef(&points_[4]), polys_[4].Evaluate(points_[4]));
    // clang-format on
  }

 protected:
  PCS pcs_;
  std::vector<Poly> polys_;
  std::vector<F> points_;
  std::vector<Commitment> commitments_;
  std::vector<PolynomialOpening<Poly>> prover_openings_;
  std::vector<PolynomialOpening<Poly, Commitment>> verifier_openings_;
};

}  // namespace

TEST_F(SHPlonkTest, CreateAndVerifyProof) {
  SHPlonkProof<Commitment> proof;
  base::Uint8VectorBuffer write_buffer;
  TranscriptWriter writer(std::move(write_buffer));
  ASSERT_TRUE(pcs_.CreateOpeningProof(prover_openings_, &proof, &writer));

  base::Buffer read_buf(writer.buffer().buffer(), writer.buffer().buffer_len());
  TranscriptReader reader(std::move(read_buf));
  EXPECT_TRUE((pcs_.VerifyOpeningProof(verifier_openings_, proof, reader)));
}

}  // namespace tachyon::crypto
