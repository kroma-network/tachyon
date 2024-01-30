// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_SELECTOR_ASSIGNMENT_H_
#define TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_SELECTOR_ASSIGNMENT_H_

#include <memory>
#include <utility>

#include "tachyon/zk/expressions/expression.h"

namespace tachyon::zk {

// This describes the assigned combination of a particular selector as well as
// the expression it should be substituted with.
template <typename F>
class SelectorAssignment {
 public:
  SelectorAssignment() = default;
  SelectorAssignment(size_t selector_index, size_t combination_index,
                     std::unique_ptr<Expression<F>> expression)
      : selector_index_(selector_index),
        combination_index_(combination_index),
        expression_(std::move(expression)) {}

  size_t selector_index() const { return selector_index_; }
  size_t combination_index() const { return combination_index_; }
  const Expression<F>* expression() const { return expression_.get(); }

 private:
  // The selector that this structure references, by index.
  size_t selector_index_ = 0;
  // The combination this selector was assigned to.
  size_t combination_index_ = 0;
  // The expression we wish to substitute with.
  std::unique_ptr<Expression<F>> expression_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_SELECTOR_ASSIGNMENT_H_
