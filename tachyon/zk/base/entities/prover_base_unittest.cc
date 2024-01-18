#include "gtest/gtest.h"

#include "tachyon/zk/plonk/halo2/prover_test.h"

namespace tachyon::zk {

namespace {

class ProverBaseTest : public halo2::ProverTest {};

}  // namespace

TEST_F(ProverBaseTest, CommitEvalsWithBlind) {
  const Domain* domain = prover_->domain();
  // setting random polynomial
  Evals evals = domain->Random<Evals>();

  // setting struct to get output
  BlindedPolynomial<Poly> out = prover_->CommitAndWriteToProofWithBlind(evals);
  EXPECT_EQ(out.poly(), prover_->domain()->IFFT(evals));
}

}  // namespace tachyon::zk
