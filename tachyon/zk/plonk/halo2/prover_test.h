#ifndef TACHYON_ZK_PLONK_HALO2_PROVER_TEST_H_
#define TACHYON_ZK_PLONK_HALO2_PROVER_TEST_H_

#include <memory>
#include <utility>

#include "gtest/gtest.h"

#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"
#include "tachyon/zk/plonk/halo2/blake2b_transcript.h"
#include "tachyon/zk/plonk/halo2/constants.h"
#include "tachyon/zk/plonk/halo2/prover.h"

namespace tachyon::zk::plonk::halo2 {

constexpr size_t kMaxDegree = (size_t{1} << 5) - 1;
constexpr size_t kMaxDomainSize = kMaxDegree + 1;
constexpr size_t kMaxExtendedDegree = (size_t{1} << 7) - 1;
constexpr size_t kMaxExtendedDomainSize = kMaxExtendedDegree + 1;

template <typename PCS, typename LS>
class ProverTest : public testing::Test {
 public:
  using F = typename PCS::Field;
  using Commitment = typename PCS::Commitment;
  using Poly = typename PCS::Poly;
  using Evals = typename PCS::Evals;
  using RationalEvals = typename PCS::RationalEvals;
  using Domain = typename PCS::Domain;
  using ExtendedDomain = typename PCS::ExtendedDomain;
  using ExtendedEvals = typename PCS::ExtendedEvals;

  void SetUp() override {
    PCS pcs;
    ASSERT_TRUE(pcs.UnsafeSetup(kMaxDomainSize));

    base::Uint8VectorBuffer write_buf;
    std::unique_ptr<crypto::TranscriptWriter<Commitment>> writer =
        std::make_unique<Blake2bWriter<Commitment>>(std::move(write_buf));

    prover_ = std::make_unique<Prover<PCS, LS>>(
        Prover<PCS, LS>::CreateFromSeed(std::move(pcs), std::move(writer),
                                        kXORShiftSeed, /*blinding_factors=*/0));
    prover_->set_domain(Domain::Create(kMaxDomainSize));
    prover_->set_extended_domain(
        ExtendedDomain::Create(kMaxExtendedDomainSize));
  }

 protected:
  Verifier<PCS, LS> CreateVerifier(base::Buffer read_buf) {
    std::unique_ptr<crypto::TranscriptReader<Commitment>> reader =
        std::make_unique<Blake2bReader<Commitment>>(std::move(read_buf));
    return prover_->ToVerifier(std::move(reader));
  }

  std::unique_ptr<Prover<PCS, LS>> prover_;
};

}  // namespace tachyon::zk::plonk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_PROVER_TEST_H_
