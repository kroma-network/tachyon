// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_REGION_H_
#define TACHYON_ZK_PLONK_CIRCUIT_REGION_H_

#include <memory>
#include <string>
#include <utility>

#include "tachyon/base/functional/callback.h"
#include "tachyon/zk/plonk/circuit/assigned_cell.h"
#include "tachyon/zk/plonk/circuit/column_key.h"
#include "tachyon/zk/plonk/circuit/selector.h"

namespace tachyon::zk {

template <typename F>
class Region {
 public:
  using AssignCallback = base::OnceCallback<Value<F>()>;

  class Layouter {
   public:
    using AssignCallback = base::OnceCallback<Value<math::RationalField<F>>()>;

    virtual ~Layouter() = default;

    // Enables a |selector| at the given |offset|.
    virtual void EnableSelector(std::string_view name, const Selector& selector,
                                size_t offset) {}

    // Allows the circuit implementer to name a Column within a Region
    // context.
    //
    // This is useful in order to improve the amount of information that
    // |prover.Verify()| and |prover.AssertSatisfied()| can provide.
    virtual void NameColumn(std::string_view name, const AnyColumnKey& column) {
    }

    // Assign an advice column value (witness)
    virtual Cell AssignAdvice(std::string_view name,
                              const AdviceColumnKey& column, size_t offset,
                              AssignCallback assign) = 0;

    // Assigns a constant value to the column |advice| at |offset| within this
    // region.
    //
    // The constant value will be assigned to a cell within one of the fixed
    // columns configured via |ConstraintSystem::EnableConstant|.
    virtual Cell AssignAdviceFromConstant(
        std::string_view name, const AdviceColumnKey& column, size_t offset,
        const math::RationalField<F>& constant) = 0;

    // Assign the value of the instance column's cell at absolute location
    // |row| to the column |advice| at |offset| within this region.
    virtual AssignedCell<F> AssignAdviceFromInstance(
        std::string_view name, const InstanceColumnKey& instance, size_t row,
        const AdviceColumnKey& advice, size_t offset) = 0;

    // Assign a fixed value
    virtual Cell AssignFixed(std::string_view name,
                             const FixedColumnKey& column, size_t offset,
                             AssignCallback assign) = 0;

    // Constrain a cell to have a constant value.
    virtual void ConstrainConstant(const Cell& cell,
                                   const math::RationalField<F>& constant) {}

    // Constrain two cells to have the same value.
    virtual void ConstrainEqual(const Cell& left, const Cell& right) {}
  };

  explicit Region(Layouter* layouter) : layouter_(layouter) {}

  // See the comment above.
  void EnableSelector(std::string_view name, const Selector& selector,
                      size_t offset) {
    layouter_->EnableSelector(name, selector, offset);
  }

  // See the comment above.
  void NameColumn(std::string_view name, const AnyColumnKey& column) {
    layouter_->NameColumn(name, column);
  }

  // See the comment above.
  AssignedCell<F> AssignAdvice(std::string_view name,
                               const AdviceColumnKey& column, size_t offset,
                               AssignCallback assign) {
    Value<F> value = Value<F>::Unknown();
    Cell cell =
        layouter_->AssignAdvice(name, column, offset, [&value, &assign]() {
          value = std::move(assign).Run();
          return value.ToRationalFieldValue();
        });
    return {std::move(cell), std::move(value)};
  }

  // See the comment above.
  AssignedCell<F> AssignAdviceFromConstant(std::string_view name,
                                           const AdviceColumnKey& column,
                                           size_t offset, const F& constant) {
    Cell cell = layouter_->AssignAdviceFromConstant(
        name, column, offset, math::RationalField<F>(constant));
    return {std::move(cell), Value<F>::Known(constant)};
  }

  // See the comment above.
  AssignedCell<F> AssignAdviceFromInstance(std::string_view name,
                                           const InstanceColumnKey& instance,
                                           size_t row,
                                           const AdviceColumnKey& advice,
                                           size_t offset) {
    return layouter_->AssignAdviceFromInstance(name, instance, row, advice,
                                               offset);
  }

  // See the comment above.
  AssignedCell<F> AssignFixed(std::string_view name,
                              const FixedColumnKey& column, size_t offset,
                              AssignCallback assign) {
    Value<F> value = Value<F>::Unknown();
    Cell cell =
        layouter_->AssignFixed(name, column, offset, [&value, &assign]() {
          value = std::move(assign).Run();
          return value.ToRationalFieldValue();
        });
    return {std::move(cell), std::move(value)};
  }

  // See the comment above.
  void ConstrainConstant(const Cell& cell,
                         const math::RationalField<F>& constant) {
    layouter_->ConstrainConstant(cell, constant);
  }

  // See the comment above.
  void ConstrainEqual(const Cell& left, const Cell& right) {
    layouter_->ConstrainEqual(left, right);
  }

 private:
  Layouter* const layouter_;
};

template <typename F>
void Selector::Enable(Region<F>& region, size_t offset) const {
  region.EnableSelector("", *this, offset);
}

template <typename F>
AssignedCell<F> AssignedCell<F>::CopyAdvice(std::string_view name,
                                            Region<F>& region,
                                            const AdviceColumnKey& column,
                                            size_t offset) const {
  AssignedCell<F> ret =
      region.AssignAdvice(name, column, offset, [this]() { return value_; });
  region.ConstrainEqual(ret.cell_, cell_);
  return ret;
}

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_REGION_H_
