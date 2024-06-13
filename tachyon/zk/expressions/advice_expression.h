// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_EXPRESSIONS_ADVICE_EXPRESSION_H_
#define TACHYON_ZK_EXPRESSIONS_ADVICE_EXPRESSION_H_

#include <memory>
#include <string>

#include "absl/memory/memory.h"

#include "tachyon/zk/expressions/expression.h"
#include "tachyon/zk/plonk/constraint_system/query.h"

namespace tachyon::zk {

template <typename F>
class AdviceExpression : public Expression<F> {
 public:
  static std::unique_ptr<AdviceExpression> CreateForTesting(
      const plonk::AdviceQuery& query) {
    return absl::WrapUnique(new AdviceExpression(query));
  }

  const plonk::AdviceQuery& query() const { return query_; }

  // Expression methods
  size_t Degree() const override { return 1; }

  uint64_t Complexity() const override { return 1; }

  std::unique_ptr<Expression<F>> Clone() const override {
    return absl::WrapUnique(new AdviceExpression(query_));
  }

  std::string ToString() const override {
    return absl::Substitute("{type: $0, column: $1}",
                            ExpressionTypeToString(this->type_),
                            query_.ToString());
  }

  void WriteIdentifier(std::ostream& out) const override {
    out << "advice[" << query_.column().index() << "]["
        << query_.rotation().value() << "]";
  }

  bool operator==(const Expression<F>& other) const override {
    if (!Expression<F>::operator==(other)) return false;
    const AdviceExpression* advice = other.ToAdvice();
    return query_ == advice->query_;
  }

 private:
  friend class ExpressionFactory<F>;

  explicit AdviceExpression(const plonk::AdviceQuery& query)
      : Expression<F>(ExpressionType::kAdvice), query_(query) {}

  plonk::AdviceQuery query_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_EXPRESSIONS_ADVICE_EXPRESSION_H_
