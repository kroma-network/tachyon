// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_LAYOUT_LOOKUP_TABLE_H_
#define TACHYON_ZK_PLONK_LAYOUT_LOOKUP_TABLE_H_

#include <string>
#include <utility>

#include "tachyon/math/base/rational_field.h"
#include "tachyon/zk/base/row_index.h"
#include "tachyon/zk/base/value.h"
#include "tachyon/zk/plonk/layout/lookup_table_column.h"

namespace tachyon::zk {

// A lookup table in the circuit.
template <typename F>
class LookupTable {
 public:
  using AssignCallback = base::OnceCallback<Value<F>()>;

  class Layouter {
   public:
    using AssignCallback = base::OnceCallback<Value<math::RationalField<F>>()>;

    virtual ~Layouter() = default;

    // Assign a fixed value to a table cell.
    //
    // Return false if the table cell has already been assigned.
    [[nodiscard]] virtual bool AssignCell(std::string_view name,
                                          const LookupTableColumn& column,
                                          RowIndex offset,
                                          AssignCallback assign) {
      return true;
    }
  };

  explicit LookupTable(Layouter* layouter) : layouter_(layouter) {}

  [[nodiscard]] bool AssignCell(std::string_view name,
                                const LookupTableColumn& column,
                                RowIndex offset, AssignCallback assign) {
    return layouter_->AssignCell(name, column, offset, [&assign]() {
      return Value<math::RationalField<F>>::Known(
          math::RationalField<F>(std::move(assign).Run().value()));
    });
  }

 private:
  // not owned
  Layouter* const layouter_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_LAYOUT_LOOKUP_TABLE_H_
