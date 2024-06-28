// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_EXPRESSIONS_INSTANCE_EXPRESSION_H_
#define TACHYON_ZK_PLONK_EXPRESSIONS_INSTANCE_EXPRESSION_H_

#include <memory>
#include <string>

#include "absl/memory/memory.h"

#include "tachyon/zk/expressions/expression.h"
#include "tachyon/zk/plonk/constraint_system/query.h"

namespace tachyon::zk::plonk {

template <typename F>
class ExpressionFactory;

template <typename F>
class InstanceExpression : public Expression<F> {
 public:
  static std::unique_ptr<InstanceExpression> CreateForTesting(
      const InstanceQuery& query) {
    return absl::WrapUnique(new InstanceExpression(query));
  }

  const InstanceQuery& query() const { return query_; }

  // Expression methods
  size_t Degree() const override { return 1; }

  uint64_t Complexity() const override { return 1; }

  std::unique_ptr<Expression<F>> Clone() const override {
    return absl::WrapUnique(new InstanceExpression(query_));
  }

  std::string ToString() const override {
    return absl::Substitute("{type: $0, column: $1}",
                            ExpressionTypeToString(this->type_),
                            query_.ToString());
  }

  bool operator==(const Expression<F>& other) const override {
    if (!Expression<F>::operator==(other)) return false;
    const InstanceExpression* instance = other.ToInstance();
    return query_ == instance->query_;
  }

 private:
  friend class ExpressionFactory<F>;

  explicit InstanceExpression(const InstanceQuery& query)
      : Expression<F>(ExpressionType::kInstance), query_(query) {}

  InstanceQuery query_;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_EXPRESSIONS_INSTANCE_EXPRESSION_H_