#ifndef TACHYON_ZK_BASE_HALO2_PROVER_TEST_H_
#define TACHYON_ZK_BASE_HALO2_PROVER_TEST_H_

#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g2.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"
#include "tachyon/zk/base/commitments/kzg_commitment_scheme_extension.h"
#include "tachyon/zk/base/halo2_prover.h"
#include "tachyon/zk/transcript/poseidon_transcript.h"

namespace tachyon::zk {

class Halo2ProverTest : public testing::Test {
 public:
  constexpr static size_t kMaxDegree = (size_t{1} << 5) - 1;
  constexpr static size_t kDomainSize = kMaxDegree + 1;
  constexpr static size_t kMaxExtendedDegree = (size_t{1} << 7) - 1;
  constexpr static size_t kExtendedDomainSize = kMaxExtendedDegree + 1;
  constexpr static size_t kBlindingFactors = 5;
  constexpr static size_t kUsableRows = kDomainSize - (kBlindingFactors + 1);

  using PCS = KZGCommitmentSchemeExtension<
      math::bn254::G1AffinePoint, math::bn254::G2AffinePoint, kMaxDegree,
      kMaxExtendedDegree, math::bn254::G1AffinePoint>;
  using F = PCS::Field;
  using Commitment = PCS::Commitment;
  using Poly = PCS::Poly;
  using Evals = PCS::Evals;
  using Domain = PCS::Domain;
  using ExtendedDomain = PCS::ExtendedDomain;

  static void SetUpTestSuite() { math::bn254::G1Curve::Init(); }

  void SetUp() override {
    PCS pcs;
    ASSERT_TRUE(pcs.UnsafeSetup(kDomainSize));

    std::unique_ptr<Domain> domain =
        math::UnivariateEvaluationDomainFactory<F, kMaxDegree>::Create(
            kDomainSize);

    std::unique_ptr<ExtendedDomain> extended_domain =
        math::UnivariateEvaluationDomainFactory<F, kMaxExtendedDegree>::Create(
            kExtendedDomainSize);

    base::VectorBuffer write_buf;
    std::unique_ptr<TranscriptWriter<math::bn254::G1AffinePoint>> writer =
        absl::WrapUnique(
            new PoseidonWriter<math::bn254::G1Curve>(std::move(write_buf)));

    prover_ = std::make_unique<Halo2Prover<PCS>>(
        Halo2Prover<PCS>::CreateFromRandomSeed(
            std::move(pcs), std::move(domain), std::move(extended_domain),
            std::move(writer), kBlindingFactors));
  }

 protected:
  std::unique_ptr<Halo2Prover<PCS>> prover_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_HALO2_PROVER_TEST_H_
