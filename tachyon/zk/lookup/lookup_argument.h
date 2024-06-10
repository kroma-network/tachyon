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
        input_expressions_(std::move(input_expressions)),
        table_expressions_(std::move(table_expressions)) {}
  Argument(std::string_view name, Pairs<std::unique_ptr<Expression<F>>> pairs)
      : name_(std::string(name)) {
    input_expressions_.reserve(pairs.size());
    table_expressions_.reserve(pairs.size());

    for (Pair<std::unique_ptr<Expression<F>>>& pair : pairs) {
      input_expressions_.push_back(std::move(pair).TakeInput());
      table_expressions_.push_back(std::move(pair).TakeTable());
    }

    pairs.clear();
  }

  const std::vector<std::unique_ptr<Expression<F>>>& input_expressions() const {
    return input_expressions_;
  }

  std::vector<std::unique_ptr<Expression<F>>>& input_expressions() {
    return input_expressions_;
  }

  const std::vector<std::unique_ptr<Expression<F>>>& table_expressions() const {
    return table_expressions_;
  }

  std::vector<std::unique_ptr<Expression<F>>>& table_expressions() {
    return table_expressions_;
  }

  bool operator==(const Argument& other) const {
    if (name_ != other.name_) return false;
    if (input_expressions_.size() != other.input_expressions_.size())
      return false;
    if (table_expressions_.size() != other.table_expressions_.size())
      return false;
    for (size_t i = 0; i < input_expressions_.size(); ++i) {
      if (*input_expressions_[i] != *other.input_expressions_[i]) return false;
    }
    for (size_t i = 0; i < table_expressions_.size(); ++i) {
      if (*table_expressions_[i] != *other.table_expressions_[i]) return false;
    }
    return true;
  }
  bool operator!=(const Argument& other) const { return !operator==(other); }

  size_t RequiredDegree() const {
    CHECK_EQ(input_expressions_.size(), table_expressions_.size());
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
    // degree (2 + max_input_degree + max_table_degree) or 4, whichever is
    // larger:
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
    size_t max_input_degree = GetMaxExprDegree(input_expressions_);

    size_t max_table_degree = GetMaxExprDegree(table_expressions_);

    // In practice because input_degree and table_degree are initialized to
    // one, the latter half of this max() invocation is at least 4 always,
    // rendering this call pointless except to be explicit in case we change
    // the initialization of input_degree/table_degree in the future.

    // NOTE(chokobole): Even though, this actually is same as |2 +
    // max_input_degree + max_table_degree|, for a better explanation, we follow
    // the Halo2 style.
    return std::max(
        // (1 - (l_last + l_blind)) * Z(ω * X) * (A'(X) + β) * (S'(X) + γ)
        size_t{4},
        // clang-format off
        // (1 - (l_last + l_blind)) * Z(X) * (A_compressed(X) + β) * (S_compressed(X) + γ)
        // clang-format on
        size_t{2} + max_input_degree + max_table_degree);
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
  std::vector<std::unique_ptr<Expression<F>>> input_expressions_;
  std::vector<std::unique_ptr<Expression<F>>> table_expressions_;
};

}  // namespace tachyon::zk::lookup

#endif  // TACHYON_ZK_LOOKUP_LOOKUP_ARGUMENT_H_
