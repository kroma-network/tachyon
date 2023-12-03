// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_EXPRESSIONS_SELECTOR_EXPRESSION_H_
#define TACHYON_ZK_EXPRESSIONS_SELECTOR_EXPRESSION_H_

#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/substitute.h"

#include "tachyon/zk/expressions/expression.h"
#include "tachyon/zk/plonk/circuit/selector.h"

namespace tachyon::zk {

template <typename F>
class SelectorExpression : public Expression<F> {
 public:
  static std::unique_ptr<SelectorExpression> CreateForTesting(
      const Selector& selector) {
    return absl::WrapUnique(new SelectorExpression(selector));
  }

  const Selector& selector() const { return selector_; }

  bool operator==(const Expression<F>& other) const {
    if (!Expression<F>::operator==(other)) return false;
    const SelectorExpression* selector = other.ToSelector();
    return selector_ == selector->selector_;
  }
  bool operator!=(const Expression<F>& other) const {
    return !operator==(other);
  }

  // Expression methods
  size_t Degree() const override { return 1; }

  uint64_t Complexity() const override { return 1; }

  std::unique_ptr<Expression<F>> Clone() const override {
    return absl::WrapUnique(new SelectorExpression(selector_));
  }

  std::string ToString() const override {
    return absl::Substitute("{type: $0, selector: $1}",
                            ExpressionTypeToString(this->type_),
                            selector_.ToString());
  }

 private:
  friend class ExpressionFactory<F>;

  explicit SelectorExpression(const Selector& selector)
      : Expression<F>(ExpressionType::kSelector), selector_(selector) {}

  Selector selector_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_EXPRESSIONS_SELECTOR_EXPRESSION_H_
