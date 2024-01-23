// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_REGION_LAYOUTER_H_
#define TACHYON_ZK_PLONK_CIRCUIT_REGION_LAYOUTER_H_

#include <stddef.h>

#include "tachyon/base/functional/callback.h"
#include "tachyon/math/base/rational_field.h"
#include "tachyon/zk/base/value.h"
#include "tachyon/zk/plonk/circuit/assigned_cell.h"
#include "tachyon/zk/plonk/circuit/column_key.h"
#include "tachyon/zk/plonk/circuit/selector.h"

namespace tachyon::zk {

template <typename F>
class RegionLayouter {
 public:
  using AssignCallback = base::OnceCallback<Value<math::RationalField<F>>()>;

  virtual ~RegionLayouter() = default;

  // Enables a |selector| at the given |offset|.
  virtual void EnableSelector(std::string_view name, const Selector& selector,
                              size_t offset) {}

  // Allows the circuit implementer to name a Column within a Region
  // context.
  //
  // This is useful in order to improve the amount of information that
  // |prover.Verify()| and |prover.AssertSatisfied()| can provide.
  virtual void NameColumn(std::string_view name, const AnyColumnKey& column) {}

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
  virtual Cell AssignFixed(std::string_view name, const FixedColumnKey& column,
                           size_t offset, AssignCallback assign) = 0;

  // Constrain a cell to have a constant value.
  virtual void ConstrainConstant(const Cell& cell,
                                 const math::RationalField<F>& constant) {}

  // Constrain two cells to have the same value.
  virtual void ConstrainEqual(const Cell& left, const Cell& right) {}
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_REGION_LAYOUTER_H_
