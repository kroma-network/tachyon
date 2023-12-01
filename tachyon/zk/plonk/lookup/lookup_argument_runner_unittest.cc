// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/lookup/lookup_argument_runner.h"

#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "tachyon/zk/plonk/circuit/expressions/expression_factory.h"
#include "tachyon/zk/plonk/lookup/compress_expression.h"
#include "tachyon/zk/plonk/lookup/permute_expression_pair.h"
#include "tachyon/zk/plonk/lookup/test/compress_expression_test_setting.h"

namespace tachyon::zk {

class LookupArgumentRunnerTest : public CompressExpressionTestSetting {};

TEST_F(LookupArgumentRunnerTest, ComputePermutationProduct) {
  constexpr size_t kBlindingFactors = 5;

  const F beta = F::Random();
  const F gamma = F::Random();

  std::vector<std::unique_ptr<Expression<F>>> input_expressions;
  std::vector<std::unique_ptr<Expression<F>>> table_expressions;
  for (size_t i = 0; i < kDomainSize; ++i) {
    F random = F::Random();
    input_expressions.push_back(ExpressionFactory<F>::Constant(random));
    table_expressions.push_back(ExpressionFactory<F>::Constant(random));
  }

  Evals compressed_input_expression;
  ASSERT_TRUE(CompressExpressions(input_expressions, prover_->domain()->size(),
                                  theta_, evaluator_,
                                  &compressed_input_expression));
  Evals compressed_table_expression;
  ASSERT_TRUE(CompressExpressions(table_expressions, prover_->domain()->size(),
                                  theta_, evaluator_,
                                  &compressed_table_expression));

  LookupPair<Evals> compressed_evals_pair(
      std::move(compressed_input_expression),
      std::move(compressed_table_expression));

  LookupPair<Evals> permuted_evals_pair;
  Error err = PermuteExpressionPair(prover_.get(), compressed_evals_pair,
                                    &permuted_evals_pair);
  ASSERT_EQ(err, Error::kNone);

  LookupPermuted<Poly, Evals> lookup_permuted(
      std::move(compressed_evals_pair), std::move(permuted_evals_pair),
      BlindedPolynomial<Poly>(), BlindedPolynomial<Poly>());

  Evals z_evals = GrandProductArgument::CreatePolynomial<Evals>(
      kDomainSize, kBlindingFactors,
      LookupArgumentRunner<Poly, Evals>::CreateNumeratorCallback<F>(
          lookup_permuted, beta, gamma),
      LookupArgumentRunner<Poly, Evals>::CreateDenominatorCallback<F>(
          lookup_permuted, beta, gamma));
  const std::vector<F>& z = z_evals.evaluations();

  // sanity check brought from halo2
  ASSERT_EQ(z[0], F::One());
  for (size_t i = 0; i < kUsableRows; ++i) {
    F left = z[i + 1];

    left *= (beta + *lookup_permuted.permuted_evals_pair().input()[i]);
    left *= (gamma + *lookup_permuted.permuted_evals_pair().table()[i]);

    F right = z[i];
    F input_term = *lookup_permuted.compressed_evals_pair().input()[i];
    F table_term = *lookup_permuted.compressed_evals_pair().table()[i];

    input_term += beta;
    table_term += gamma;
    right *= input_term * table_term;

    ASSERT_EQ(left, right);
  }
}

}  // namespace tachyon::zk
