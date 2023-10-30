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

#include "tachyon/zk/plonk/circuit/expressions/expression.h"

namespace tachyon::zk {

template <typename F>
class LookupArgument {
 public:
  struct TableMapElem {
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
      input_expressions_->push_back(std::move(elem.input));
      table_expressions_->push_back(std::move(elem.table));
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

 private:
  std::string name_;
  std::vector<std::unique_ptr<Expression<F>>> input_expressions_;
  std::vector<std::unique_ptr<Expression<F>>> table_expressions_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_LOOKUP_LOOKUP_ARGUMENT_H_
