// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_LOOKUP_LOOKUP_ARGUMENT_H_
#define TACHYON_ZK_LOOKUP_LOOKUP_ARGUMENT_H_

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tachyon/zk/expressions/expression.h"
#include "tachyon/zk/lookup/lookup_pair.h"

namespace tachyon::zk::lookup {

template <typename F>
class Argument {
 public:
  Argument() = default;
  Argument(std::string_view name,
           std::vector<std::unique_ptr<Expression<F>>> input_expressions,
           std::vector<std::unique_ptr<Expression<F>>> table_expressions)
      : name_(std::string(name)),
        table_expressions_(std::move(table_expressions)) {
    inputs_expressions_.push_back(std::move(input_expressions));
  }
  Argument(std::string_view name,
           std::vector<std::vector<std::unique_ptr<Expression<F>>>>
               inputs_expressions,
           std::vector<std::unique_ptr<Expression<F>>> table_expressions)
      : name_(std::string(name)),
        inputs_expressions_(std::move(inputs_expressions)),
        table_expressions_(std::move(table_expressions)) {}
  Argument(std::string_view name, Pairs<std::unique_ptr<Expression<F>>> pairs)
      : name_(std::string(name)) {
    std::vector<std::unique_ptr<Expression<F>>> input_expressions;
    input_expressions.reserve(pairs.size());
    table_expressions_.reserve(pairs.size());

    for (Pair<std::unique_ptr<Expression<F>>>& pair : pairs) {
      input_expressions.push_back(std::move(pair).TakeInput());
      table_expressions_.push_back(std::move(pair).TakeTable());
    }
    inputs_expressions_.push_back(std::move(input_expressions));

    pairs.clear();
  }

  const std::vector<std::vector<std::unique_ptr<Expression<F>>>>&
  inputs_expressions() const {
    return inputs_expressions_;
  }

  std::vector<std::vector<std::unique_ptr<Expression<F>>>>&
  inputs_expressions() {
    return inputs_expressions_;
  }

  const std::vector<std::unique_ptr<Expression<F>>>& input_expressions() const {
    return inputs_expressions_[0];
  }

  std::vector<std::unique_ptr<Expression<F>>>& input_expressions() {
    return inputs_expressions_[0];
  }

  const std::vector<std::unique_ptr<Expression<F>>>& table_expressions() const {
    return table_expressions_;
  }

  std::vector<std::unique_ptr<Expression<F>>>& table_expressions() {
    return table_expressions_;
  }

  bool operator==(const Argument& other) const {
    if (name_ != other.name_) return false;
    if (inputs_expressions_.size() != other.inputs_expressions_.size())
      return false;
    if (table_expressions_.size() != other.table_expressions_.size())
      return false;
    for (size_t i = 0; i < inputs_expressions_.size(); ++i) {
      if (inputs_expressions_[i].size() != other.inputs_expressions_[i].size())
        return false;
      for (size_t j = 0; j < inputs_expressions_[i].size(); ++j) {
        if (*inputs_expressions_[i][j] != *other.inputs_expressions_[i][j])
          return false;
      }
    }
    for (size_t i = 0; i < table_expressions_.size(); ++i) {
      if (*table_expressions_[i] != *other.table_expressions_[i]) return false;
    }
    return true;
  }
  bool operator!=(const Argument& other) const { return !operator==(other); }

  size_t RequiredDegree() const {
    for (const std::vector<std::unique_ptr<Expression<F>>>& input_expressions :
         inputs_expressions_) {
      CHECK_EQ(input_expressions.size(), table_expressions_.size());
    }
    // [Halo2 Lookup]
    // See https://zcash.github.io/halo2/design/proving-system/lookup.html
    // for more details.
    //
    // The first value in the permutation poly should be one.
    // degree 2:
    // l_first(X) * (1 - Z(X)) = 0
    //
    // The "last" value in the permutation poly should be a boolean, for
    // completeness and soundness.
    //
    // degree 3:
    // l_last(X) * (Z(X)² - Z(X)) = 0
    //
    // Enable the permutation argument for only the rows involved.
    // degree (2 + |combined_input_degree| + |max_table_degree|) or 4, whichever
    // is larger:
    // clang-format off
    // (1 - (l_last(X) + l_blind(X))) * (Z(ω * X) * (A'(X) + β) * (S'(X) + γ) - Z(X) * (A_compressed(X) + β) * (S_compressed(X) + γ)) = 0
    // clang-format on
    //
    // The first two values of A' and S' should be the same.
    //
    // degree 2:
    // l_first(X) * (A'(X) - S'(X)) = 0
    //
    // Either the two values are the same, or the previous value of A' is the
    // same as the current value.
    //
    // degree 3:
    // clang-format off
    // (1 - (l_last(X) + l_blind(X))) * (A′(X) − S′(X)) * (A′(X) − A′(ω⁻¹ * X)) = 0
    // clang-format on
    //
    // [LogDerivativeHalo2]
    // The first value in the sum poly should be zero.
    // degree 2:
    // l_first(X) * ϕ(X) = 0
    //
    // The last value in the sum poly should be zero.
    // degree 2:
    // l_last(X) * ϕ(X) = 0
    //
    // Enable the sum argument for only the rows involved.
    // degree (2 + |combined_input_degree| + |max_table_degree|) or
    // (3 + |inputs_expressions_.size()|), whichever is larger:
    // clang-format off
    // φᵢ(X) = fᵢ(X) + β
    // τ(X) = t(X) + β
    // LHS = τ(X) * Π(φᵢ(X)) * (ϕ(ω * X) - ϕ(X))
    //     ↪ DEG(LHS) = |max_table_degree| + |combined_input_degree| + 1
    // RHS = τ(X) * Π(φᵢ(X)) * ((Σ 1/φᵢ(X)) - m(X) / τ(X))
    //     ↪ DEG(RHS) = |combined_input_degree| + 1
    // (1 - (l_last(X) + l_blind(X))) * (ϕ(ω * X) * τ(X) * Π(φᵢ(X)) - (LHS - RHS))
    // clang-format on
    size_t combined_input_degree = std::accumulate(
        inputs_expressions_.begin(), inputs_expressions_.end(), 0,
        [](size_t combined_degree,
           const std::vector<std::unique_ptr<Expression<F>>>&
               input_expressions) {
          return combined_degree + GetMaxExprDegree(input_expressions);
        });

    size_t max_table_degree = GetMaxExprDegree(table_expressions_);

    // In practice because input_degree and table_degree are initialized to
    // one, the latter half of this max() invocation is at least 4 always,
    // rendering this call pointless except to be explicit in case we change
    // the initialization of input_degree/table_degree in the future.
    return std::max(
        // clang-format off
        // [Halo2 Lookup]
        // NOTE(Insun35): The size of |inputs_expressions_| for Halo2 lookup is 1.
        // Thus, (3 + |inputs_expressions_.size()|) is always 4.
        // (1 - (l_last(X) + l_blind(X))) * Z(ω * X) * (A'(X) + β) * (S'(X) + γ)
        //   ↪ degree = 4
        //
        // [LogDerivativeHalo2]
        // (1 - (l_last(X) + l_blind(X))) * ϕ(ω * X) * τ(X) * Π(φᵢ(X))
        //   ↪ degree = |3 + inputs_expressions_.size()|
        // clang-format on
        3 + inputs_expressions_.size(),
        // clang-format off
        // [Halo2 Lookup]
        // max_degree =
        //    DEG((l_last(X) + l_blind(X))) * Z(X) * (A_compressed(X) + β) * (S_compressed(X) + γ))
        //
        // [LogDerivativeHalo2]
        // LHS = τ(X) * Π(φᵢ(X)) * (ϕ(ω * X) - ϕ(X))
        //     ↪ DEG(LHS) = |max_table_degree| + |combined_input_degree| + 1
        // RHS = τ(X) * Π(φᵢ(X)) * ((Σ 1/φᵢ(X)) - m(X) / τ(X))
        //     ↪ DEG(RHS) = |combined_input_degree| + 1
        // max_degree = DEG((1 - (l_last(X) + l_blind(X))) * (LHS - RHS))
        //            = 1 + DEG(LHS) = 2 + |combined_input_degree| + |max_table_degree|
        // clang-format on
        size_t{2} + combined_input_degree + max_table_degree);
  }

 private:
  static size_t GetMaxExprDegree(
      const std::vector<std::unique_ptr<Expression<F>>>& expressions) {
    return std::accumulate(
        expressions.begin(), expressions.end(), 1,
        [](size_t degree, const std::unique_ptr<Expression<F>>& expr_ptr) {
          return std::max(degree, expr_ptr->Degree());
        });
  }

  std::string name_;
  std::vector<std::vector<std::unique_ptr<Expression<F>>>> inputs_expressions_;
  std::vector<std::unique_ptr<Expression<F>>> table_expressions_;
};

}  // namespace tachyon::zk::lookup

#endif  // TACHYON_ZK_LOOKUP_LOOKUP_ARGUMENT_H_
