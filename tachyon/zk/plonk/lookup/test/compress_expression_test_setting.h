#ifndef TACHYON_ZK_PLONK_LOOKUP_TEST_COMPRESS_EXPRESSION_TEST_SETTING_H_
#define TACHYON_ZK_PLONK_LOOKUP_TEST_COMPRESS_EXPRESSION_TEST_SETTING_H_

#include <memory>
#include <vector>

#include "tachyon/zk/base/halo2_prover_test.h"
#include "tachyon/zk/plonk/circuit/expressions/evaluator/simple_evaluator.h"

namespace tachyon::zk {

class CompressExpressionTestSetting : public Halo2ProverTest {
 public:
  void SetUp() override {
    Halo2ProverTest::SetUp();

    SimpleEvaluator<Evals>::Arguments arguments(
        &advice_values_, &fixed_values_, &instance_values_, &challenges_);
    evaluator_ = {0, static_cast<int32_t>(prover_->domain()->size()), 1,
                  arguments};
    theta_ = F(2);
  }

 protected:
  SimpleEvaluator<Evals> evaluator_;
  std::vector<Evals> advice_values_;
  std::vector<Evals> fixed_values_;
  std::vector<Evals> instance_values_;
  std::vector<F> challenges_;
  F theta_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_LOOKUP_TEST_COMPRESS_EXPRESSION_TEST_SETTING_H_
