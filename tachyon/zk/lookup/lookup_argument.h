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

namespace tachyon::zk {

template <typename F>
class LookupArgument {
 public:
  LookupArgument() = default;
  LookupArgument(std::string_view name,
                 std::vector<std::unique_ptr<Expression<F>>> input_expressions,
                 std::vector<std::unique_ptr<Expression<F>>> table_expressions)
      : name_(std::string(name)),
        input_expressions_(std::move(input_expressions)),
        table_expressions_(std::move(table_expressions)) {}
  LookupArgument(std::string_view name,
                 LookupPairs<std::unique_ptr<Expression<F>>> pairs)
      : name_(std::string(name)) {
    input_expressions_.reserve(pairs.size());
    table_expressions_.reserve(pairs.size());

    for (LookupPair<std::unique_ptr<Expression<F>>>& pair : pairs) {
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

  bool operator==(const LookupArgument& other) const {
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
  bool operator!=(const LookupArgument& other) const {
    return !operator==(other);
  }

  size_t RequiredDegree() const {
    CHECK_EQ(input_expressions_.size(), table_expressions_.size());

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

 private:
  std::string name_;
  std::vector<std::unique_ptr<Expression<F>>> input_expressions_;
  std::vector<std::unique_ptr<Expression<F>>> table_expressions_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_LOOKUP_LOOKUP_ARGUMENT_H_
