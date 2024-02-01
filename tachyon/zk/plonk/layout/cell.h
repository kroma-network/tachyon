// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_LAYOUT_CELL_H_
#define TACHYON_ZK_PLONK_LAYOUT_CELL_H_

#include <stddef.h>

#include <string>

#include "absl/strings/substitute.h"

#include "tachyon/export.h"
#include "tachyon/zk/base/row_index.h"
#include "tachyon/zk/plonk/base/column_key.h"

namespace tachyon::zk::plonk {

// A pointer to a cell within a circuit.
class TACHYON_EXPORT Cell {
 public:
  Cell() = default;
  Cell(size_t region_index, RowIndex row_offset, const AnyColumnKey& column)
      : region_index_(region_index), row_offset_(row_offset), column_(column) {}

  size_t region_index() const { return region_index_; }
  RowIndex row_offset() const { return row_offset_; }
  const AnyColumnKey& column() const { return column_; }

  std::string ToString() const {
    return absl::Substitute("{region_index: $0, row_offset: $1, column: $2}",
                            region_index_, row_offset_, column_.ToString());
  }

 private:
  // Identifies the region in which this cell resides.
  size_t region_index_ = 0;
  // The relative offset of this cell within its region.
  RowIndex row_offset_ = 0;
  // The column of this cell.
  AnyColumnKey column_;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_LAYOUT_CELL_H_
