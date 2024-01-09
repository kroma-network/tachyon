// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_ALLOCATIONS_H_
#define TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_ALLOCATIONS_H_

#include <stddef.h>

#include <algorithm>
#include <vector>

#include "absl/container/btree_set.h"

#include "tachyon/export.h"
#include "tachyon/zk/plonk/circuit/floor_planner/allocated_region.h"
#include "tachyon/zk/plonk/circuit/floor_planner/empty_space.h"

namespace tachyon::zk {

// Allocated rows within a column.

// This is a set of |AllocatedRegion|s, representing disjoint allocated
// intervals.
class TACHYON_EXPORT Allocations {
 public:
  absl::btree_set<AllocatedRegion>& allocations() { return allocations_; }

  // Returns the row that forms the unbounded unallocated interval.
  size_t UnboundedIntervalStart() const {
    if (allocations_.empty()) {
      return 0;
    }
    return allocations_.rbegin()->End();
  }

  // Return all the *unallocated* non-empty intervals intersecting [|start|,
  // |end|). |end| = std::nullopt represents an unbounded end.
  std::vector<EmptySpace> FreeIntervals(size_t start,
                                        std::optional<size_t> end) const {
    std::vector<EmptySpace> result;
    result.reserve(allocations_.size() + 1);
    size_t row = start;
    for (const AllocatedRegion& region : allocations_) {
      if (end.has_value() && region.start() >= end.value()) {
        break;
      }
      if (row < region.start()) {
        result.emplace_back(row, std::optional<size_t>(region.start()));
      }
      row = std::max(row, region.End());
    }
    if (!end.has_value() || row < end.value()) {
      result.emplace_back(row, end);
    }
    return result;
  }

 private:
  absl::btree_set<AllocatedRegion> allocations_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_ALLOCATIONS_H_
