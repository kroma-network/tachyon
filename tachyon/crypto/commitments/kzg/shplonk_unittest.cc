#include "tachyon/crypto/commitments/kzg/shplonk.h"

#include <memory>

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g2.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"
#include "tachyon/zk/plonk/halo2/poseidon_transcript.h"

namespace tachyon::crypto {

namespace {

class SHPlonkTest : public testing::Test {
 public:
  constexpr static size_t K = 3;
  constexpr static size_t N = size_t{1} << K;
  constexpr static size_t kMaxDegree = N - 1;

  using PCS = SHPlonk<math::bn254::G1AffinePoint, math::bn254::G2AffinePoint,
                      kMaxDegree, math::bn254::G1AffinePoint>;
  using F = PCS::Field;
  using Poly = PCS::Poly;
  using Commitment = PCS::Commitment;
  using Point = Poly::Point;
  using PolyRef = base::DeepRef<const Poly>;
  using PointRef = base::DeepRef<const Point>;

  static void SetUpTestSuite() { math::bn254::G1Curve::Init(); }

  SHPlonkTest() : writer_(base::Uint8VectorBuffer()) {}

  void SetUp() override {
    KZG<math::bn254::G1AffinePoint, kMaxDegree, math::bn254::G1AffinePoint> kzg;
    pcs_ = PCS(std::move(kzg), &writer_);
    ASSERT_TRUE(pcs_.UnsafeSetup(N));

    polys_ = base::CreateVector(5, []() { return Poly::Random(kMaxDegree); });
    points_ = {F(1), F(2), F(3), F(4), F(5)};

    poly_openings_.reserve(8);
    // {P₀, [x₀, x₁, x₂]}
    poly_openings_.emplace_back(PolyRef(&polys_[0]), PointRef(&points_[0]),
                                polys_[0].Evaluate(points_[0]));
    poly_openings_.emplace_back(PolyRef(&polys_[0]), PointRef(&points_[1]),
                                polys_[0].Evaluate(points_[1]));
    poly_openings_.emplace_back(PolyRef(&polys_[0]), PointRef(&points_[2]),
                                polys_[0].Evaluate(points_[2]));
    // (P₁, [x₀, x₁, x₂]}
    poly_openings_.emplace_back(PolyRef(&polys_[1]), PointRef(&points_[0]),
                                polys_[1].Evaluate(points_[0]));
    poly_openings_.emplace_back(PolyRef(&polys_[1]), PointRef(&points_[1]),
                                polys_[1].Evaluate(points_[1]));
    poly_openings_.emplace_back(PolyRef(&polys_[1]), PointRef(&points_[2]),
                                polys_[1].Evaluate(points_[2]));
    // (P₂, [x₀, x₁, x₂]}
    poly_openings_.emplace_back(PolyRef(&polys_[2]), PointRef(&points_[0]),
                                polys_[2].Evaluate(points_[0]));
    poly_openings_.emplace_back(PolyRef(&polys_[2]), PointRef(&points_[1]),
                                polys_[2].Evaluate(points_[1]));
    poly_openings_.emplace_back(PolyRef(&polys_[2]), PointRef(&points_[2]),
                                polys_[2].Evaluate(points_[2]));
    // (P₃, [x₂, x₃]}
    poly_openings_.emplace_back(PolyRef(&polys_[3]), PointRef(&points_[2]),
                                polys_[3].Evaluate(points_[2]));
    poly_openings_.emplace_back(PolyRef(&polys_[3]), PointRef(&points_[3]),
                                polys_[3].Evaluate(points_[3]));
    // (P₄, [x₃, x₄]}
    poly_openings_.emplace_back(PolyRef(&polys_[4]), PointRef(&points_[3]),
                                polys_[4].Evaluate(points_[3]));
    poly_openings_.emplace_back(PolyRef(&polys_[4]), PointRef(&points_[4]),
                                polys_[4].Evaluate(points_[4]));
  }

 protected:
  PCS pcs_;
  std::vector<Poly> polys_;
  std::vector<F> points_;
  std::vector<PolynomialOpening<Poly>> poly_openings_;
  zk::halo2::PoseidonWriter<Commitment> writer_;
};

}  // namespace

TEST_F(SHPlonkTest, CreateProof) {
  SHPlonkProof<Commitment> proof;
  ASSERT_TRUE(pcs_.CreateOpeningProof(poly_openings_, &proof));
}

}  // namespace tachyon::crypto
