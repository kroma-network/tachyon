// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_EXPRESSIONS_EXPRESSION_FACTORY_H_
#define TACHYON_ZK_PLONK_EXPRESSIONS_EXPRESSION_FACTORY_H_

#include <memory>
#include <utility>

#include "absl/memory/memory.h"

#include "tachyon/zk/expressions/expression_factory.h"
#include "tachyon/zk/plonk/expressions/advice_expression.h"
#include "tachyon/zk/plonk/expressions/challenge_expression.h"
#include "tachyon/zk/plonk/expressions/fixed_expression.h"
#include "tachyon/zk/plonk/expressions/instance_expression.h"
#include "tachyon/zk/plonk/expressions/selector_expression.h"

namespace tachyon::zk::plonk {

template <typename F>
class ExpressionFactory : public tachyon::zk::ExpressionFactory<F> {
 public:
  using Expr = Expression<F>;

  static std::unique_ptr<Expr> Advice(const AdviceQuery& query) {
    return absl::WrapUnique(new AdviceExpression<F>(query));
  }

  static std::unique_ptr<Expr> Instance(const InstanceQuery& query) {
    return absl::WrapUnique(new InstanceExpression<F>(query));
  }

  static std::unique_ptr<Expr> Challenge(Challenge challenge) {
    return absl::WrapUnique(new ChallengeExpression<F>(challenge));
  }

  static std::unique_ptr<Expr> Fixed(const FixedQuery& query) {
    return absl::WrapUnique(new FixedExpression<F>(query));
  }

  static std::unique_ptr<Expr> Selector(Selector selector) {
    return absl::WrapUnique(new SelectorExpression<F>(selector));
  }
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_EXPRESSIONS_EXPRESSION_FACTORY_H_
