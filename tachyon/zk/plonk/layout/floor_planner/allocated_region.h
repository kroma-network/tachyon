// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_LAYOUT_FLOOR_PLANNER_ALLOCATED_REGION_H_
#define TACHYON_ZK_PLONK_LAYOUT_FLOOR_PLANNER_ALLOCATED_REGION_H_

#include "tachyon/export.h"
#include "tachyon/zk/base/row_types.h"

namespace tachyon::zk::plonk {

// A region allocated within a column.
class TACHYON_EXPORT AllocatedRegion {
 public:
  constexpr AllocatedRegion(RowIndex start, RowIndex length)
      : start_(start), length_(length) {}

  constexpr RowIndex start() const { return start_; }
  constexpr RowIndex length() const { return length_; }

  constexpr RowIndex End() const { return start_ + length_; }

  constexpr bool operator<(const AllocatedRegion& other) const {
    return start_ < other.start_;
  }

 private:
  // The starting position of the region.
  RowIndex start_ = 0;
  // The length of the region.
  RowIndex length_ = 0;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_LAYOUT_FLOOR_PLANNER_ALLOCATED_REGION_H_
