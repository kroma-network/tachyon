#ifndef TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_VIRTUAL_CELLS_H_
#define TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_VIRTUAL_CELLS_H_

#include <memory>
#include <utility>
#include <vector>

#include "tachyon/zk/expressions/expression_factory.h"
#include "tachyon/zk/plonk/constraint_system/selector.h"
#include "tachyon/zk/plonk/constraint_system/virtual_cell.h"

namespace tachyon::zk::plonk {

template <typename F>
class ConstraintSystem;

// Exposes the "virtual cells" that can be queried while creating a custom gate
// or lookup table.
template <typename F>
class VirtualCells {
 public:
  explicit VirtualCells(ConstraintSystem<F>* meta) : meta_(meta) {}

  const std::vector<Selector>& queried_selectors() const& {
    return queried_selectors_;
  }
  const std::vector<VirtualCell>& queried_cells() const& {
    return queried_cells_;
  }

  std::vector<Selector>&& TakeQueriedSelectors() && {
    return std::move(queried_selectors_);
  }
  std::vector<VirtualCell>&& TakeQueriedCells() && {
    return std::move(queried_cells_);
  }

  // Query a selector at the current position.
  std::unique_ptr<Expression<F>> QuerySelector(const Selector& selector) {
    queried_selectors_.push_back(selector);
    return ExpressionFactory<F>::Selector(selector);
  }

  // Query a fixed column at a relative position
  std::unique_ptr<Expression<F>> QueryFixed(const FixedColumnKey& column,
                                            Rotation at) {
    queried_cells_.emplace_back(column, at);
    return ExpressionFactory<F>::Fixed(
        {meta_->QueryFixedIndex(column, at), at, column});
  }

  // Query an advice column at a relative position
  std::unique_ptr<Expression<F>> QueryAdvice(const AdviceColumnKey& column,
                                             Rotation at) {
    queried_cells_.emplace_back(column, at);
    return ExpressionFactory<F>::Advice(
        {meta_->QueryAdviceIndex(column, at), at, column});
  }

  // Query an instance column at a relative position
  std::unique_ptr<Expression<F>> QueryInstance(const InstanceColumnKey& column,
                                               Rotation at) {
    queried_cells_.emplace_back(column, at);
    return ExpressionFactory<F>::Instance(
        {meta_->QueryInstanceIndex(column, at), at, column});
  }

  // Query an Any column at a relative position
  std::unique_ptr<Expression<F>> QueryAny(const AnyColumnKey& column,
                                          Rotation at) {
    switch (column.type()) {
      case ColumnType::kAdvice:
        return QueryAdvice(column, at);
      case ColumnType::kFixed:
        return QueryFixed(column, at);
      case ColumnType::kInstance:
        return QueryInstance(column, at);
      case ColumnType::kAny:
        break;
    }
    NOTREACHED();
    return nullptr;
  }

  // Query a challenge
  std::unique_ptr<Expression<F>> QueryChallenge(const Challenge& challenge) {
    return ExpressionFactory<F>::Challenge(challenge);
  }

 private:
  // not owned
  ConstraintSystem<F>* const meta_;
  std::vector<Selector> queried_selectors_;
  std::vector<VirtualCell> queried_cells_;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_VIRTUAL_CELLS_H_
