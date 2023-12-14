#include "gtest/gtest.h"

#include "tachyon/zk/plonk/halo2/halo2_prover_test.h"

namespace tachyon::zk {

namespace {

class ProverTest : public Halo2ProverTest {};

}  // namespace

TEST_F(ProverTest, CommitEvalsWithBlind) {
  const Domain* domain = prover_->domain();
  // setting random polynomial
  Evals evals = domain->Random<Evals>();

  // setting struct to get output
  BlindedPolynomial<Poly> out;
  ASSERT_TRUE(prover_->CommitEvalsWithBlind(evals, &out));

  EXPECT_EQ(out.poly(), prover_->domain()->IFFT(evals));
}

}  // namespace tachyon::zk
