#ifndef TACHYON_ZK_PLONK_CIRCUIT_VIRTUAL_CELLS_H_
#define TACHYON_ZK_PLONK_CIRCUIT_VIRTUAL_CELLS_H_

#include <memory>
#include <utility>
#include <vector>

#include "tachyon/zk/plonk/circuit/expressions/expression_factory.h"
#include "tachyon/zk/plonk/circuit/selector.h"
#include "tachyon/zk/plonk/circuit/virtual_cell.h"

namespace tachyon::zk {

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

  std::vector<Selector>&& queried_selectors() && {
    return std::move(queried_selectors_);
  }

  const std::vector<VirtualCell>& queried_cells() const& {
    return queried_cells_;
  }

  std::vector<VirtualCell>&& queried_cells() && {
    return std::move(queried_cells_);
  }

  // Query a selector at the current position.
  std::unique_ptr<Expression<F>> QuerySelector(const Selector& selector) {
    queried_selectors_.push_back(selector);
    return ExpressionFactory::Selector(selector);
  }

  // Query a fixed column at a relative position
  std::unique_ptr<Expression<F>> QueryFixed(const FixedColumn& column,
                                            Rotation at) {
    queried_cells_.push_back({column, at});
    return ExpressionFactory::Fixed({
        meta_->QueryFixedIndex(column, at),
        column.index,
        at,
    });
  }

  // Query an advice column at a relative position
  std::unique_ptr<Expression<F>> QueryAdvice(const AdviceColumn& column,
                                             Rotation at) {
    queried_cells_.push_back({column, at});
    return ExpressionFactory::Advice({
        meta_->QueryAdviceIndex(column, at),
        column.index,
        at,
        column.type().phase,
    });
  }

  // Query an instance column at a relative position
  std::unique_ptr<Expression<F>> QueryInstance(const InstanceColumn& column,
                                               Rotation at) {
    .queried_cells_.push_back({column, at});
    return ExpressionFactory::Instance({
        meta_->QueryInstanceIndex(column, at),
        column.index,
        at,
    });
  }

  // Query an Any column at a relative position
  std::unique_ptr<Expression<F>> QueryAny(const AnyColumn& column,
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
    return ExpressionFactory::Challenge(challenge);
  }

 private:
  ConstraintSystem<F>* const meta_;
  std::vector<Selector> queried_selectors_;
  std::vector<VirtualCell> queried_cells_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_VIRTUAL_CELLS_H_
