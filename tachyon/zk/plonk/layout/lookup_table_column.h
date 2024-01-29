// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_LAYOUT_LOOKUP_TABLE_COLUMN_H_
#define TACHYON_ZK_PLONK_LAYOUT_LOOKUP_TABLE_COLUMN_H_

#include <utility>

#include "absl/hash/hash.h"

#include "tachyon/export.h"
#include "tachyon/zk/plonk/base/column_key.h"

namespace tachyon::zk {

// A fixed column of a lookup table.
//
// A lookup table can be loaded into this column via
// |Layouter::AssignLookupTable()]. Columns can currently only contain a single
// table, but they may be used in multiple lookup arguments via
// |ConstraintSystem::Lookup()|.
//
// Lookup table columns are always "encumbered" by the lookup arguments they are
// used in; they cannot simultaneously be used as general fixed columns.
class TACHYON_EXPORT LookupTableColumn {
 public:
  LookupTableColumn() = default;
  explicit LookupTableColumn(const FixedColumnKey& column) : column_(column) {}

  // The fixed column that this table column is stored in.
  //
  // # Security
  //
  // This inner column MUST NOT be exposed in the public API, or else chip
  // developers can load lookup tables into their circuits without
  // default-value-filling the columns, which can cause soundness bugs.
  const FixedColumnKey& column() const { return column_; }

  bool operator==(const LookupTableColumn& column) const {
    return column_ == column.column_;
  }
  bool operator!=(const LookupTableColumn& column) const {
    return column_ != column.column_;
  }

 private:
  FixedColumnKey column_;
};

template <typename H>
H AbslHashValue(H h, const LookupTableColumn& column) {
  return H::combine(std::move(h), column.column());
}

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_LAYOUT_LOOKUP_TABLE_COLUMN_H_
