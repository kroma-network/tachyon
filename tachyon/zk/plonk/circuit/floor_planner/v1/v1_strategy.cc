// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/circuit/floor_planner/v1/v1_strategy.h"

#include <stdint.h>

namespace tachyon::zk {

// - |start| is the current start row of the region (not of this column).
// - |slack| is the maximum number of rows the start could be moved down,
// taking into account prior columns.
std::optional<RowIndex> FirstFitRegion(
    CircuitAllocations* column_allocations,
    const std::vector<RegionColumn>& region_columns, RowIndex region_length,
    RowIndex start, std::optional<RowIndex> slack) {
  if (region_columns.empty()) {
    return start;
  }

  const RegionColumn& c = region_columns[0];
  std::vector<RegionColumn> remaining_columns(region_columns.begin() + 1,
                                              region_columns.end());

  std::optional<RowIndex> end;
  if (slack.has_value()) {
    end = start + region_length + slack.value();
  }

  // Iterate over the unallocated non-empty intervals in |c| that intersect
  // [start, end).
  for (const EmptySpace& space :
       column_allocations->try_emplace(c, Allocations())
           .first->second.FreeIntervals(start, end)) {
    // Do we have enough room for this column of the region in this interval?
    std::optional<int32_t> s_slack;
    if (space.end().has_value()) {
      s_slack = space.end().value() - space.start() - region_length;
    }
    if (slack.has_value() && s_slack.has_value()) {
      CHECK_LE(s_slack.value(), static_cast<int64_t>(slack.value()));
    }
    if (!s_slack.has_value() || s_slack.value() >= 0) {
      std::optional<RowIndex> row = FirstFitRegion(
          column_allocations, remaining_columns, region_length, space.start(),
          s_slack.has_value() ? std::optional<RowIndex>(s_slack.value())
                              : std::nullopt);
      if (row.has_value()) {
        if (end.has_value()) {
          CHECK_LE(row.value() + region_length, end.value());
        }
        column_allocations->at(c).allocations().insert(
            AllocatedRegion(row.value(), region_length));
        return row;
      }
    }
  }

  // No placement worked; the caller will need to try other possibilities.
  return std::nullopt;
}

}  // namespace tachyon::zk
