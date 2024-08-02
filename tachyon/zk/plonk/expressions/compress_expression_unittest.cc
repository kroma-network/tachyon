// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/expressions/compress_expression.h"

#include "gtest/gtest.h"

#include "tachyon/zk/expressions/expression_factory.h"
#include "tachyon/zk/plonk/expressions/test/compress_expression_test.h"

namespace tachyon::zk::plonk {

TEST_F(CompressExpressionTest, CompressExpressions) {
  const size_t kExpressionSize = 10;
  std::vector<F> values =
      base::CreateVector(kExpressionSize, [](size_t i) { return F::Random(); });

  // setting |expressions| to be compressed
  std::vector<std::unique_ptr<Expression<F>>> expressions =
      base::CreateVector(kExpressionSize, [&values](size_t i) {
        return ExpressionFactory<F>::Constant(values[i]);
      });

  size_t n = prover_->pcs().N();
  ProvingEvaluator<Evals> evaluator = evaluator_;
  std::vector<F, base::memory::ReusingAllocator<F>> expected(n);
  for (size_t i = 0; i < expressions.size(); ++i) {
    F value = evaluator.Evaluate(expressions[i].get());
    for (size_t j = 0; j < n; ++j) {
      expected[j] *= theta_;
      expected[j] += value;
    }
  }

  Evals out =
      CompressExpressions(prover_->domain(), expressions, theta_, evaluator_);
  EXPECT_EQ(out, Evals(std::move(expected)));
}

}  // namespace tachyon::zk::plonk
