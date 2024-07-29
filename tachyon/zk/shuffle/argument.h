// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_SHUFFLE_ARGUMENT_H_
#define TACHYON_ZK_SHUFFLE_ARGUMENT_H_

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tachyon/zk/expressions/expression.h"

namespace tachyon::zk::shuffle {

template <typename F>
class Argument {
 public:
  Argument() = default;
  Argument(std::string_view name,
           std::vector<std::unique_ptr<Expression<F>>> input_expressions,
           std::vector<std::unique_ptr<Expression<F>>> shuffle_expressions)
      : name_(std::string(name)),
        input_expressions_(std::move(input_expressions)),
        shuffle_expressions_(std::move(shuffle_expressions)) {}

  const std::vector<std::unique_ptr<Expression<F>>>& input_expressions() const {
    return input_expressions_;
  }

  std::vector<std::unique_ptr<Expression<F>>>& input_expressions() {
    return input_expressions_;
  }

  const std::vector<std::unique_ptr<Expression<F>>>& shuffle_expressions()
      const {
    return shuffle_expressions_;
  }

  std::vector<std::unique_ptr<Expression<F>>>& shuffle_expressions() {
    return shuffle_expressions_;
  }

  bool operator==(const Argument& other) const {
    if (name_ != other.name_) return false;
    if (input_expressions_.size() != other.input_expressions_.size())
      return false;
    if (shuffle_expressions_.size() != other.shuffle_expressions_.size())
      return false;
    for (size_t i = 0; i < input_expressions_.size(); ++i) {
      if (*input_expressions_[i] != *other.input_expressions_[i]) return false;
    }
    for (size_t i = 0; i < shuffle_expressions_.size(); ++i) {
      if (*shuffle_expressions_[i] != *other.shuffle_expressions_[i])
        return false;
    }
    return true;
  }
  bool operator!=(const Argument& other) const { return !operator==(other); }

  size_t RequiredDegree() const {
    size_t max_input_degree = GetMaxExprDegree(input_expressions_);
    size_t max_shuffle_degree = GetMaxExprDegree(shuffle_expressions_);

    // clang-format off
    // (1 - (l_last(X) + l_blind(X))) * (z(ω * X) * (s(X) + γ) - z(X) * (a(X) + γ))
    // clang-format on
    return 2 + std::max(max_shuffle_degree, max_input_degree);
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
  std::vector<std::unique_ptr<Expression<F>>> shuffle_expressions_;
};

}  // namespace tachyon::zk::shuffle

#endif  // TACHYON_ZK_SHUFFLE_ARGUMENT_H_
