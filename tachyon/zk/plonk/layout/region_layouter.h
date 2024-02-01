// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_LAYOUT_REGION_LAYOUTER_H_
#define TACHYON_ZK_PLONK_LAYOUT_REGION_LAYOUTER_H_

#include "tachyon/base/functional/callback.h"
#include "tachyon/math/base/rational_field.h"
#include "tachyon/zk/base/row_index.h"
#include "tachyon/zk/base/value.h"
#include "tachyon/zk/plonk/base/column_key.h"
#include "tachyon/zk/plonk/constraint_system/selector.h"
#include "tachyon/zk/plonk/layout/assigned_cell.h"

namespace tachyon::zk::plonk {

template <typename F>
class RegionLayouter {
 public:
  using AssignCallback = base::OnceCallback<Value<math::RationalField<F>>()>;

  virtual ~RegionLayouter() = default;

  // Enables a |selector| at the given |offset|.
  virtual void EnableSelector(std::string_view name, const Selector& selector,
                              RowIndex offset) {}

  // Allows the circuit implementer to name a Column within a Region
  // context.
  //
  // This is useful in order to improve the amount of information that
  // |prover.Verify()| and |prover.AssertSatisfied()| can provide.
  virtual void NameColumn(std::string_view name, const AnyColumnKey& column) {}

  // Assign an advice column value (witness)
  virtual Cell AssignAdvice(std::string_view name,
                            const AdviceColumnKey& column, RowIndex offset,
                            AssignCallback assign) = 0;

  // Assigns a constant value to the column |advice| at |offset| within this
  // region.
  //
  // The constant value will be assigned to a cell within one of the fixed
  // columns configured via |ConstraintSystem::EnableConstant|.
  virtual Cell AssignAdviceFromConstant(
      std::string_view name, const AdviceColumnKey& column, RowIndex offset,
      const math::RationalField<F>& constant) = 0;

  // Assign the value of the instance column's cell at absolute location
  // |row| to the column |advice| at |offset| within this region.
  virtual AssignedCell<F> AssignAdviceFromInstance(
      std::string_view name, const InstanceColumnKey& instance, RowIndex row,
      const AdviceColumnKey& advice, RowIndex offset) = 0;

  // Assign a fixed value
  virtual Cell AssignFixed(std::string_view name, const FixedColumnKey& column,
                           RowIndex offset, AssignCallback assign) = 0;

  // Constrain a cell to have a constant value.
  virtual void ConstrainConstant(const Cell& cell,
                                 const math::RationalField<F>& constant) {}

  // Constrain two cells to have the same value.
  virtual void ConstrainEqual(const Cell& left, const Cell& right) {}
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_LAYOUT_REGION_LAYOUTER_H_
