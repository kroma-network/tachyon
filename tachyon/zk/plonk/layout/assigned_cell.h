// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_LAYOUT_ASSIGNED_CELL_H_
#define TACHYON_ZK_PLONK_LAYOUT_ASSIGNED_CELL_H_

#include <string>
#include <utility>

#include "tachyon/zk/base/row_types.h"
#include "tachyon/zk/base/value.h"
#include "tachyon/zk/plonk/layout/cell.h"

namespace tachyon::zk::plonk {

template <typename F>
class Region;

// An assigned cell.
template <typename F>
class AssignedCell {
 public:
  AssignedCell() = default;
  AssignedCell(const Cell& cell, const Value<F>& value)
      : cell_(cell), value_(value) {}
  AssignedCell(const Cell& cell, Value<F>&& value)
      : cell_(cell), value_(std::move(value)) {}

  const Cell& cell() const { return cell_; }
  const Value<F>& value() const { return value_; }

  // Copies the value to a given advice cell and constrains them to be equal.
  AssignedCell<F> CopyAdvice(std::string_view, Region<F>& region,
                             const AdviceColumnKey& column,
                             RowIndex offset) const;

  std::string ToString() const {
    return absl::Substitute("{cell: $0, value: $1}", cell_.ToString(),
                            value_.ToString());
  }

 private:
  Cell cell_;
  Value<F> value_;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_LAYOUT_ASSIGNED_CELL_H_
