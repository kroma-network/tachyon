// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_LAYOUT_FLOOR_PLANNER_EMPTY_SPACE_H_
#define TACHYON_ZK_PLONK_LAYOUT_FLOOR_PLANNER_EMPTY_SPACE_H_

#include <optional>

#include "tachyon/base/range.h"
#include "tachyon/export.h"
#include "tachyon/zk/base/row_index.h"

namespace tachyon::zk::plonk {

// An area of empty space within a column.
class TACHYON_EXPORT EmptySpace {
 public:
  EmptySpace(RowIndex start, std::optional<RowIndex> end)
      : start_(start), end_(end) {}

  RowIndex start() const { return start_; }
  std::optional<RowIndex> end() const { return end_; }

  constexpr std::optional<base::Range<RowIndex>> Range() const {
    if (end_.has_value()) {
      return std::optional<base::Range<RowIndex>>(
          base::Range<RowIndex>(start_, end_.value()));
    }
    return std::nullopt;
  }

 private:
  // The starting position (inclusive) of the empty space.
  RowIndex start_ = 0;
  // The ending position (exclusive) of the empty space, or |std::nullopt| if
  // unbounded.
  std::optional<RowIndex> end_ = std::nullopt;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_LAYOUT_FLOOR_PLANNER_EMPTY_SPACE_H_
