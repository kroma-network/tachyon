// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_REGION_H_
#define TACHYON_ZK_PLONK_CIRCUIT_REGION_H_

#include <memory>
#include <string>

#include "tachyon/base/functional/callback.h"
#include "tachyon/math/base/rational_field.h"
#include "tachyon/zk/plonk/circuit/assigned_cell.h"
#include "tachyon/zk/plonk/circuit/column.h"
#include "tachyon/zk/plonk/circuit/selector.h"
#include "tachyon/zk/plonk/error.h"

namespace tachyon::zk {

template <typename F>
class Region {
 public:
  using AnnotateCallback = base::OnceCallback<std::string()>;
  using AssignCallback = base::OnceCallback<Value<math::RationalField<F>>()>;

  class Layouter {
   public:
    virtual ~Layouter() = default;

    // Enables a |selector| at the given |offset|.
    virtual Error EnableSelector(AnnotateCallback annotate,
                                 const Selector& selector, size_t offset) {
      return Error::kNone;
    }

    // Allows the circuit implementer to name/annotate a Column within a Region
    // context.
    //
    // This is useful in order to improve the amount of information that
    // |prover.Verify()| and |prover.AssertSatisfied()| can provide.
    virtual void NameColumn(AnnotateCallback annotate,
                            const AnyColumn& column) {}

    // Assign an advice column value (witness)
    virtual Error AssignAdvice(AnnotateCallback annotate,
                               const AdviceColumn& column, size_t offset,
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
        AnnotateCallback annotate, const AdviceColumn& column, size_t offset,
        const math::RationalField<F>& constant, Cell* cell) {
      return Error::kNone;
    }

    // Assign the value of the instance column's cell at absolute location
    // |row| to the column |advice| at |offset| within this region.
    //
    // Returns |Error::kNone| and populates |cell| if known.
    virtual Error AssignAdviceFromInstance(
        AnnotateCallback annotate, const InstanceColumn& instance, size_t row,
        const AdviceColumn& advice, size_t offset, AssignedCell<F>* cell) {
      return Error::kNone;
    }

    // Assign a fixed value
    virtual Error AssignFixed(AnnotateCallback annotate,
                              const FixedColumn& column, size_t offset,
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
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_REGION_H_
