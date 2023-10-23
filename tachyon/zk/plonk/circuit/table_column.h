// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_TABLE_COLUMN_H_
#define TACHYON_ZK_PLONK_CIRCUIT_TABLE_COLUMN_H_

#include "tachyon/export.h"
#include "tachyon/zk/plonk/circuit/column.h"

namespace tachyon::zk {

// A fixed column of a lookup table.
//
// A lookup table can be loaded into this column via |Layouter::AssignTable()].
// Columns can currently only contain a single table, but they may be used in
// multiple lookup arguments via |ConstraintSystem::Lookup()|.
//
// Lookup table columns are always "encumbered" by the lookup arguments they are
// used in; they cannot simultaneously be used as general fixed columns.
class TACHYON_EXPORT TableColumn {
  TableColumn() = default;
  explicit TableColumn(const FixedColumn& column) : column_(column) {}

  // The fixed column that this table column is stored in.
  //
  // # Security
  //
  // This inner column MUST NOT be exposed in the public API, or else chip
  // developers can load lookup tables into their circuits without
  // default-value-filling the columns, which can cause soundness bugs.
  const FixedColumn& column() const { return column_; }

 private:
  FixedColumn column_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_TABLE_COLUMN_H_
