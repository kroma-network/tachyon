// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_EMPTY_SPACE_H_
#define TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_EMPTY_SPACE_H_

#include <stddef.h>

#include <optional>

#include "tachyon/base/range.h"
#include "tachyon/export.h"

namespace tachyon::zk {

// An area of empty space within a column.
class TACHYON_EXPORT EmptySpace {
 public:
  EmptySpace(size_t start, std::optional<size_t> end)
      : start_(start), end_(end) {}

  size_t start() const { return start_; }
  std::optional<size_t> end() const { return end_; }

  constexpr std::optional<base::Range<size_t>> Range() const {
    if (end_.has_value()) {
      return std::optional<base::Range<size_t>>(
          base::Range<size_t>(start_, end_.value()));
    }
    return std::nullopt;
  }

 private:
  // The starting position (inclusive) of the empty space.
  size_t start_ = 0;
  // The ending position (exclusive) of the empty space, or |std::nullopt| if
  // unbounded.
  std::optional<size_t> end_ = std::nullopt;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_EMPTY_SPACE_H_
