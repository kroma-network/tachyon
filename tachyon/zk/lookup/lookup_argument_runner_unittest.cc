// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/lookup/lookup_argument_runner.h"

#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "tachyon/zk/expressions/expression_factory.h"
#include "tachyon/zk/lookup/compress_expression.h"
#include "tachyon/zk/lookup/permute_expression_pair.h"
#include "tachyon/zk/lookup/test/compress_expression_test_setting.h"

namespace tachyon::zk {

class LookupArgumentRunnerTest : public CompressExpressionTestSetting {};

TEST_F(LookupArgumentRunnerTest, ComputePermutationProduct) {
  prover_->blinder().set_blinding_factors(5);

  const F beta = F::Random();
  const F gamma = F::Random();

  std::vector<std::unique_ptr<Expression<F>>> input_expressions;
  std::vector<std::unique_ptr<Expression<F>>> table_expressions;
  size_t n = prover_->pcs().N();
  for (size_t i = 0; i < n; ++i) {
    F random = F::Random();
    input_expressions.push_back(ExpressionFactory<F>::Constant(random));
    table_expressions.push_back(ExpressionFactory<F>::Constant(random));
  }

  Evals compressed_input_expression = CompressExpressions(
      prover_->domain(), input_expressions, theta_, evaluator_);
  Evals compressed_table_expression = CompressExpressions(
      prover_->domain(), table_expressions, theta_, evaluator_);

  LookupPair<Evals> compressed_evals_pair(
      std::move(compressed_input_expression),
      std::move(compressed_table_expression));

  LookupPair<Evals> permuted_evals_pair;
  ASSERT_TRUE(PermuteExpressionPair(prover_.get(), compressed_evals_pair,
                                    &permuted_evals_pair));

  LookupPermuted<Poly, Evals> lookup_permuted(
      std::move(compressed_evals_pair), std::move(permuted_evals_pair),
      BlindedPolynomial<Poly>(), BlindedPolynomial<Poly>());

  Evals z_evals = plonk::GrandProductArgument::CreatePolynomial<Evals>(
      n, prover_->blinder().blinding_factors(),
      LookupArgumentRunner<Poly, Evals>::CreateNumeratorCallback(
          lookup_permuted, beta, gamma),
      LookupArgumentRunner<Poly, Evals>::CreateDenominatorCallback(
          lookup_permuted, beta, gamma));
  const std::vector<F>& z = z_evals.evaluations();

  // sanity check brought from halo2
  ASSERT_EQ(z[0], F::One());
  for (RowIndex i = 0; i < prover_->GetUsableRows(); ++i) {
    F left = z[i + 1];

    left *= (beta + lookup_permuted.permuted_evals_pair().input()[i]);
    left *= (gamma + lookup_permuted.permuted_evals_pair().table()[i]);

    F right = z[i];
    F input_term = lookup_permuted.compressed_evals_pair().input()[i];
    F table_term = lookup_permuted.compressed_evals_pair().table()[i];

    input_term += beta;
    table_term += gamma;
    right *= input_term * table_term;

    ASSERT_EQ(left, right);
  }
}

}  // namespace tachyon::zk
