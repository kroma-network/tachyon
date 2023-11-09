// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_LOOKUP_LOOKUP_ARGUMENT_H_
#define TACHYON_ZK_PLONK_LOOKUP_LOOKUP_ARGUMENT_H_

#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tachyon/zk/plonk/lookup/compress_expression.h"
#include "tachyon/zk/plonk/lookup/lookup_permuted.h"

namespace tachyon::zk {

template <typename F>
class LookupArgument {
 public:
  struct TableMapElem {
    TableMapElem() = default;
    TableMapElem(std::unique_ptr<Expression<F>> input,
                 std::unique_ptr<Expression<F>> table)
        : input(std::move(input)), table(std::move(table)) {}

    std::unique_ptr<Expression<F>> input;
    std::unique_ptr<Expression<F>> table;
  };

  using TableMap = std::vector<TableMapElem>;

  LookupArgument() = default;
  LookupArgument(const std::string_view& name, TableMap table_map)
      : name_(std::string(name)) {
    input_expressions_.reserve(table_map.size());
    table_expressions_.reserve(table_map.size());

    for (TableMapElem& elem : table_map) {
      input_expressions_.push_back(std::move(elem.input));
      table_expressions_.push_back(std::move(elem.table));
    }

    table_map.clear();
  }

  const std::vector<std::unique_ptr<Expression<F>>>& input_expressions() const {
    return input_expressions_;
  }

  const std::vector<std::unique_ptr<Expression<F>>>& table_expressions() const {
    return table_expressions_;
  }

  size_t RequiredDegree() const {
    CHECK_EQ(input_expressions_->size(), table_expressions_->size());

    size_t max_input_degree = std::accumulate(
        input_expressions_.begin(), input_expressions_.end(), 1,
        [](size_t degree, const std::unique_ptr<Expression<F>>& input_expr) {
          return std::max(degree, input_expr->Degree());
        });

    size_t max_table_degree = std::accumulate(
        table_expressions_.begin(), table_expressions_.end(), 1,
        [](size_t degree, const std::unique_ptr<Expression<F>>& table_expr) {
          return std::max(degree, table_expr->Degree());
        });

    return 2 + max_input_degree + max_table_degree;
  }

  template <typename ProverTy, typename Evals,
            typename PCSTy = typename ProverTy::PCSTy>
  LookupPermuted<PCSTy> CommitPermuted(
      ProverTy& prover, const F& theta,
      const SimpleEvaluator<Evals>& evaluator_tpl) {
    // A_compressed(X) = θᵐ⁻¹A₀(X) + θᵐ⁻²A₁(X) + ... + θAₘ₋₂(X) + Aₘ₋₁(X)
    Evals compressed_input_expression;
    CHECK(CompressExpressions(input_expressions_, prover.domain()->size(),
                              theta, evaluator_tpl,
                              &compressed_input_expression));

    // S_compressed(X) = θᵐ⁻¹S₀(X) + θᵐ⁻²S₁(X) + ... + θSₘ₋₂(X) + Sₘ₋₁(X)
    Evals compressed_table_expression;
    CHECK(CompressExpressions(table_expressions_, prover.domain()->size(),
                              theta, evaluator_tpl,
                              &compressed_table_expression));

    // Permute compressed (InputExpression, TableExpression) pair.
    EvalsPair<Evals> compressed_evals_pair(
        std::move(compressed_input_expression),
        std::move(compressed_table_expression));

    // A'(X), S'(X)
    EvalsPair<Evals> permuted_evals_pair;
    Error err = PermuteExpressionPair(prover, compressed_evals_pair,
                                      &permuted_evals_pair);
    CHECK_EQ(err, Error::kNone);

    // Commit(A'(X))
    BlindedPolynomial<PCSTy> permuted_input_poly;
    CHECK(prover.CommitEvalsWithBlind(permuted_evals_pair.input(),
                                      &permuted_input_poly));

    // Commit(S'(X))
    BlindedPolynomial<PCSTy> permuted_table_poly;
    CHECK(prover.CommitEvalsWithBlind(permuted_evals_pair.table(),
                                      &permuted_table_poly));

    return {std::move(compressed_evals_pair), std::move(permuted_evals_pair),
            std::move(permuted_input_poly), std::move(permuted_table_poly)};
  }

 private:
  std::string name_;
  std::vector<std::unique_ptr<Expression<F>>> input_expressions_;
  std::vector<std::unique_ptr<Expression<F>>> table_expressions_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_LOOKUP_LOOKUP_ARGUMENT_H_
