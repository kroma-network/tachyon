#ifndef TACHYON_ZK_PLONK_EXPRESSIONS_TEST_COMPRESS_EXPRESSION_TEST_H_
#define TACHYON_ZK_PLONK_EXPRESSIONS_TEST_COMPRESS_EXPRESSION_TEST_H_

#include <memory>
#include <vector>

#include "tachyon/zk/plonk/expressions/proving_evaluator.h"
#include "tachyon/zk/plonk/halo2/bn254_shplonk_prover_test.h"

namespace tachyon::zk::plonk {

class CompressExpressionTest : public halo2::BN254SHPlonkProverTest {
 public:
  void SetUp() override {
    halo2::BN254SHPlonkProverTest::SetUp();

    MultiPhaseRefTable<Evals> table(fixed_columns_, advice_columns_,
                                    instance_columns_, challenges_);
    evaluator_ = {0, static_cast<int32_t>(prover_->domain()->size()), 1, table};
    theta_ = F(2);
  }

 protected:
  ProvingEvaluator<Evals> evaluator_;
  std::vector<Evals> fixed_columns_;
  std::vector<Evals> advice_columns_;
  std::vector<Evals> instance_columns_;
  std::vector<F> challenges_;
  F theta_;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_EXPRESSIONS_TEST_COMPRESS_EXPRESSION_TEST_H_
