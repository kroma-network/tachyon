#ifndef TACHYON_CRYPTO_COMMITMENTS_KZG_KZG_FAMILY_TEST_H_
#define TACHYON_CRYPTO_COMMITMENTS_KZG_KZG_FAMILY_TEST_H_

#include <stddef.h>

#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "tachyon/crypto/commitments/kzg/kzg.h"
#include "tachyon/crypto/commitments/test/bn254_kzg_polynomial_openings.h"
#include "tachyon/crypto/transcripts/simple_transcript.h"
#include "tachyon/math/elliptic_curves/bn/bn254/bn254.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"

namespace tachyon::crypto {

constexpr static size_t K = 4;
constexpr static size_t N = size_t{1} << K;
constexpr static size_t kMaxDegree = N - 1;

template <typename PCS>
class KZGFamilyTest : public testing::Test {
 public:
  using F = typename PCS::Field;
  using Poly = typename PCS::Poly;
  using Commitment = typename PCS::Commitment;
  using Point = typename Poly::Point;

  static void SetUpTestSuite() { math::bn254::BN254Curve::Init(); }

  void SetUp() override {
    KZG<math::bn254::G1AffinePoint, kMaxDegree, Commitment> kzg;
    pcs_ = PCS(std::move(kzg));
    ASSERT_TRUE(pcs_.UnsafeSetup(N, F(2)));
  }

  void CreateAndVerifyProof() {
    OwnedPolynomialOpenings<Poly, Commitment> owned_openings;
    std::string error;
    ASSERT_TRUE(
        LoadAndParseJson(base::FilePath("tachyon/crypto/commitments/test/"
                                        "bn254_kzg_polynomial_openings.json"),
                         &owned_openings, &error));
    ASSERT_TRUE(error.empty());

    SimpleTranscriptWriter<Commitment> writer((base::Uint8VectorBuffer()));
    std::vector<PolynomialOpening<Poly>> prover_openings =
        owned_openings.CreateProverOpenings();
    ASSERT_TRUE(pcs_.CreateOpeningProof(prover_openings, &writer));

    base::Buffer read_buf(writer.buffer().buffer(),
                          writer.buffer().buffer_len());
    SimpleTranscriptReader<Commitment> reader(std::move(read_buf));
    std::vector<PolynomialOpening<Poly, Commitment>> verifier_openings =
        owned_openings.CreateVerifierOpenings();
    EXPECT_TRUE((pcs_.VerifyOpeningProof(verifier_openings, &reader)));
  }

  void Copyable() {
    std::vector<uint8_t> vec;
    vec.resize(base::EstimateSize(pcs_));
    base::Buffer write_buf(vec.data(), vec.size());
    ASSERT_TRUE(write_buf.Write(pcs_));
    ASSERT_TRUE(write_buf.Done());

    write_buf.set_buffer_offset(0);

    PCS value;
    ASSERT_TRUE(write_buf.Read(&value));

    EXPECT_EQ(pcs_.kzg().g1_powers_of_tau(), value.kzg().g1_powers_of_tau());
    EXPECT_EQ(pcs_.kzg().g1_powers_of_tau_lagrange(),
              value.kzg().g1_powers_of_tau_lagrange());
    EXPECT_EQ(pcs_.s_g2(), value.s_g2());
  }

 protected:
  PCS pcs_;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_KZG_KZG_FAMILY_TEST_H_
