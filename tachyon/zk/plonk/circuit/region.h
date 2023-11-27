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
#include "tachyon/zk/plonk/error.h"

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
    virtual Error EnableSelector(std::string_view name,
                                 const Selector& selector, size_t offset) {
      return Error::kNone;
    }

    // Allows the circuit implementer to name a Column within a Region
    // context.
    //
    // This is useful in order to improve the amount of information that
    // |prover.Verify()| and |prover.AssertSatisfied()| can provide.
    virtual void NameColumn(std::string_view name, const AnyColumnKey& column) {
    }

    // Assign an advice column value (witness)
    virtual Error AssignAdvice(std::string_view name,
                               const AdviceColumnKey& column, size_t offset,
                               AssignCallback assign, Cell* cell) {
      return Error::kNone;
    }

    // Assigns a constant value to the column |advice| at |offset| within this
    // region.
    //
    // The constant value will be assigned to a cell within one of the fixed
    // columns configured via |ConstraintSystem::EnableConstant|.
    //
    // Returns |Error::kNone| and populates |cell| that has been
    // equality-constrained to the constant.
    virtual Error AssignAdviceFromConstant(
        std::string_view name, const AdviceColumnKey& column, size_t offset,
        const math::RationalField<F>& constant, Cell* cell) {
      return Error::kNone;
    }

    // Assign the value of the instance column's cell at absolute location
    // |row| to the column |advice| at |offset| within this region.
    //
    // Returns |Error::kNone| and populates |cell| if known.
    virtual Error AssignAdviceFromInstance(
        std::string_view name, const InstanceColumnKey& instance, size_t row,
        const AdviceColumnKey& advice, size_t offset, AssignedCell<F>* cell) {
      return Error::kNone;
    }

    // Assign a fixed value
    virtual Error AssignFixed(std::string_view name,
                              const FixedColumnKey& column, size_t offset,
                              AssignCallback assign, Cell* cell) {
      return Error::kNone;
    }

    // Constrain a cell to have a constant value.
    //
    // Returns an error if the cell is in a column where equality has not been
    // enabled.
    virtual Error ConstrainConstant(const Cell& cell,
                                    const math::RationalField<F>& constant) {
      return Error::kNone;
    }

    // Constrain two cells to have the same value.
    //
    // Returns an error if either of the cells is not within the given
    // permutation.
    virtual Error ConstrainEqual(const Cell& left, const Cell& right) {
      return Error::kNone;
    }
  };

  explicit Region(Layouter* layouter) : layouter_(layouter) {}

  // See the comment above.
  Error EnableSelector(std::string_view name, const Selector& selector,
                       size_t offset) {
    return layouter_->EnableSelector(name, selector, offset);
  }

  // See the comment above.
  void NameColumn(std::string_view name, const AnyColumnKey& column) {
    layouter_->NameColumn(name, column);
  }

  // See the comment above.
  Error AssignAdvice(std::string_view name, const AdviceColumnKey& column,
                     size_t offset, AssignCallback assign,
                     AssignedCell<F>* assigned_cell) {
    Cell cell;
    Value<F> value = Value<F>::Unknown();
    Error error = layouter_->AssignAdvice(
        name, column, offset,
        [&value, &assign]() {
          value = std::move(assign).Run();
          return value.ToRationalFieldValue();
        },
        &cell);
    if (error != Error::kNone) return error;
    *assigned_cell = {cell, std::move(value)};
    return Error::kNone;
  }

  // See the comment above.
  Error AssignAdviceFromConstant(std::string_view name,
                                 const AdviceColumnKey& column, size_t offset,
                                 const F& constant,
                                 AssignedCell<F>* assigned_cell) {
    Cell cell;
    Error error = layouter_->AssignAdviceFromConstant(
        name, column, offset, math::RationalField<F>(constant), &cell);
    if (error != Error::kNone) return error;
    *assigned_cell = {cell, Value<F>::Known(constant)};
    return Error::kNone;
  }

  // See the comment above.
  Error AssignAdviceFromInstance(std::string_view name,
                                 const InstanceColumnKey& instance, size_t row,
                                 const AdviceColumnKey& advice, size_t offset,
                                 AssignedCell<F>* cell) {
    return layouter_->AssignAdviceFromInstance(name, instance, row, advice,
                                               offset, cell);
  }

  // See the comment above.
  Error AssignFixed(std::string_view name, const FixedColumnKey& column,
                    size_t offset, AssignCallback assign,
                    AssignedCell<F>* assigned_cell) {
    Cell cell;
    Value<F> value = Value<F>::Unknown();
    Error error = layouter_->AssignFixed(
        name, column, offset,
        [&value, &assign]() {
          value = std::move(assign).Run();
          return value.ToRationalFieldValue();
        },
        &cell);
    if (error != Error::kNone) return error;
    *assigned_cell = {cell, std::move(value)};
    return Error::kNone;
  }

  // See the comment above.
  Error ConstrainConstant(const Cell& cell,
                          const math::RationalField<F>& constant) {
    return layouter_->ConstrainConstant(cell, constant);
  }

  // See the comment above.
  Error ConstrainEqual(const Cell& left, const Cell& right) {
    return layouter_->ConstrainEqual(left, right);
  }

 private:
  Layouter* const layouter_;
};

template <typename F>
Error Selector::Enable(Region<F>& region, size_t offset) const {
  return region.EnableSelector("", *this, offset);
}

template <typename F>
Error AssignedCell<F>::CopyAdvice(std::string_view name, Region<F>& region,
                                  const AdviceColumnKey& column, size_t offset,
                                  AssignedCell<F>* assigned_cell) const {
  Error error = region.AssignAdvice(
      name, column, offset, [this]() { return value_; }, assigned_cell);
  if (error != Error::kNone) return error;
  region.ConstrainEqual(assigned_cell->cell_, cell_);
  return Error::kNone;
}

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_REGION_H_
