// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/lookup/compress_expression.h"

#include "gtest/gtest.h"

#include "tachyon/zk/plonk/circuit/expressions/expression_factory.h"
#include "tachyon/zk/plonk/lookup/test/compress_expression_test_setting.h"

namespace tachyon::zk {

class CompressExpressionTest : public CompressExpressionTestSetting {};

TEST_F(CompressExpressionTest, CompressExpressions) {
  const size_t kExpressionSize = 10;
  std::vector<F> values =
      base::CreateVector(kExpressionSize, [](size_t i) { return F::Random(); });

  // setting |expressions| to be compressed
  std::vector<std::unique_ptr<Expression<F>>> expressions =
      base::CreateVector(kExpressionSize, [&values](size_t i) {
        return ExpressionFactory<F>::Constant(values[i]);
      });

  SimpleEvaluator<Evals> evaluator = evaluator_;
  std::vector<F> expected = base::CreateVector(kDomainSize, F::Zero());
  for (size_t i = 0; i < expressions.size(); ++i) {
    F value = evaluator.Evaluate(expressions[i].get());
    for (size_t j = 0; j < kDomainSize; ++j) {
      expected[j] *= theta_;
      expected[j] += value;
    }
  }

  Evals out;
  ASSERT_TRUE(
      CompressExpressions(expressions, kDomainSize, theta_, evaluator_, &out));
  EXPECT_EQ(out, Evals(std::move(expected)));
}

}  // namespace tachyon::zk
