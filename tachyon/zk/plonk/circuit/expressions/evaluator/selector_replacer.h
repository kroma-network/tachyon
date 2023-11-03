// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_EVALUATOR_SELECTOR_REPLACER_H_
#define TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_EVALUATOR_SELECTOR_REPLACER_H_

#include <memory>
#include <vector>

#include "tachyon/zk/plonk/circuit/expressions/evaluator.h"
#include "tachyon/zk/plonk/circuit/expressions/selector_expression.h"

namespace tachyon::zk {

template <typename F>
class SelectorsReplacer : public Evaluator<F, std::unique_ptr<Expression<F>>> {
 public:
  SelectorsReplacer(
      const std::vector<std::unique_ptr<Expression<F>>>& replacements,
      bool must_be_non_simple)
      : replacements_(replacements), must_be_non_simple_(must_be_non_simple) {}

  // Evaluator methods
  std::unique_ptr<Expression<F>> Evaluate(const Expression<F>* input) override {
    if (input->type() == ExpressionType::kSelector) {
      const Selector& selector = input->ToSelector()->selector();
      if (must_be_non_simple_) {
        // Simple selectors are prohibited from appearing in
        // expressions in the lookup argument by |ConstraintSystem|.
        CHECK(!selector.is_simple());
      }
      return replacements_[selector.index()]->Clone();
    }
    return input->Clone();
  }

 private:
  const std::vector<std::unique_ptr<Expression<F>>>& replacements_;
  const bool must_be_non_simple_;
};

template <typename F>
std::unique_ptr<Expression<F>> Expression<F>::ReplaceSelectors(
    const std::vector<std::unique_ptr<Expression<F>>>& replacements,
    bool must_be_non_simple) const {
  SelectorsReplacer<F> replacer(replacements, must_be_non_simple);
  return Evaluate(&replacer);
}

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_EVALUATOR_SELECTOR_REPLACER_H_
