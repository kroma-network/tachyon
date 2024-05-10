// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_PERMUTATION_LABEL_H_
#define TACHYON_ZK_PLONK_PERMUTATION_LABEL_H_

#include <stddef.h>

#include <string>

#include "absl/strings/substitute.h"

#include "tachyon/export.h"
#include "tachyon/zk/base/row_types.h"

namespace tachyon::zk::plonk {

struct TACHYON_EXPORT Label {
  size_t col = 0;
  RowIndex row = 0;

  constexpr Label(size_t col, RowIndex row) : col(col), row(row) {}

  bool operator==(const Label& other) const {
    return col == other.col && row == other.row;
  }
  bool operator!=(const Label& other) const { return !operator==(other); }

  std::string ToString() const {
    return absl::Substitute("($0, $1)", col, row);
  }
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_LABEL_H_
