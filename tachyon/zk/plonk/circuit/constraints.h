#ifndef TACHYON_ZK_PLONK_CIRCUIT_CONSTRAINTS_H_
#define TACHYON_ZK_PLONK_CIRCUIT_CONSTRAINTS_H_

#include <memory>
#include <utility>
#include <vector>

#include "tachyon/zk/plonk/circuit/constraint.h"
#include "tachyon/zk/plonk/circuit/expressions/expression.h"

namespace tachyon::zk {

// A set of polynomial constraints with a common selector.
template <typename F>
class Constraints {
 public:
  Constraints(std::unique_ptr<Expression<F>> selector,
              const std::vector<Constraint>& constraints)
      : selector_(selector), constraints_(constraints) {}
  Constraints(std::unique_ptr<Expression<F>> selector,
              std::vector<Constraint>&& constraints)
      : selector_(selector), constraints_(std::move(constraints)) {}

  const std::unique_ptr<Expression<F>>& selector() const& { return selector_; }

  const std::vector<Constraint>& constraints() const& { return constraints_; }

 private:
  std::unique_ptr<Expression<F>> selector_;
  std::vector<Constraint> constraints_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_CONSTRAINTS_H_
