#ifndef TACHYON_ZK_PLONK_HALO2_PROVER_TEST_H_
#define TACHYON_ZK_PLONK_HALO2_PROVER_TEST_H_

#include <memory>
#include <utility>

#include "gtest/gtest.h"

#include "tachyon/crypto/transcripts/blake2b_transcript.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g2.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"
#include "tachyon/zk/base/commitments/shplonk_extension.h"
#include "tachyon/zk/plonk/halo2/constants.h"
#include "tachyon/zk/plonk/halo2/prover.h"

namespace tachyon::zk::halo2 {

class ProverTest : public testing::Test {
 public:
  constexpr static size_t kMaxDegree = (size_t{1} << 5) - 1;
  constexpr static size_t kMaxDomainSize = kMaxDegree + 1;
  constexpr static size_t kMaxExtendedDegree = (size_t{1} << 7) - 1;
  constexpr static size_t kMaxExtendedDomainSize = kMaxExtendedDegree + 1;

  using PCS = SHPlonkExtension<math::bn254::G1AffinePoint,
                               math::bn254::G2AffinePoint, kMaxDegree,
                               kMaxExtendedDegree, math::bn254::G1AffinePoint>;
  using F = PCS::Field;
  using Commitment = PCS::Commitment;
  using Poly = PCS::Poly;
  using Evals = PCS::Evals;
  using RationalEvals = PCS::RationalEvals;
  using Domain = PCS::Domain;
  using ExtendedDomain = PCS::ExtendedDomain;
  using ExtendedEvals = PCS::ExtendedEvals;

  static void SetUpTestSuite() { math::bn254::G1Curve::Init(); }

  void SetUp() override {
    PCS pcs;
    ASSERT_TRUE(pcs.UnsafeSetup(kMaxDomainSize));

    base::VectorBuffer write_buf;
    std::unique_ptr<crypto::TranscriptWriter<math::bn254::G1AffinePoint>>
        writer = std::make_unique<crypto::Blake2bWriter<math::bn254::G1Curve>>(
            std::move(write_buf), kTranscriptStr);

    constexpr uint8_t kSeed[] = {0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d,
                                 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32,
                                 0x54, 0x06, 0xbc, 0xe5};

    prover_ = std::make_unique<Prover<PCS>>(Prover<PCS>::CreateFromSeed(
        std::move(pcs), std::move(writer), kSeed, /*blinding_factors=*/0));
    prover_->set_domain(Domain::Create(kMaxDomainSize));
    prover_->set_extended_domain(
        ExtendedDomain::Create(kMaxExtendedDomainSize));
  }

 protected:
  std::unique_ptr<Prover<PCS>> prover_;
};

}  // namespace tachyon::zk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_PROVER_TEST_H_
