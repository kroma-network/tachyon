#include "gtest/gtest.h"

#include "tachyon/zk/base/halo2/halo2_prover_test.h"

namespace tachyon::zk {

namespace {

class ProverTest : public Halo2ProverTest {};

}  // namespace

TEST_F(ProverTest, CommitEvalsWithBlind) {
  // setting random polynomial
  Evals evals = Evals::Random(prover_->pcs().N() - 1);

  // setting struct to get output
  BlindedPolynomial<Poly> out;
  ASSERT_TRUE(prover_->CommitEvalsWithBlind(evals, &out));

  EXPECT_EQ(out.poly(), prover_->domain()->IFFT(evals));
}

}  // namespace tachyon::zk
