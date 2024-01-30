// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_LAYOUT_FLOOR_PLANNER_V1_V1_STRATEGY_H_
#define TACHYON_ZK_PLONK_LAYOUT_FLOOR_PLANNER_V1_V1_STRATEGY_H_

#include <stddef.h>

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/export.h"
#include "tachyon/zk/plonk/base/column_key.h"
#include "tachyon/zk/plonk/base/column_type.h"
#include "tachyon/zk/plonk/layout/floor_planner/allocations.h"
#include "tachyon/zk/plonk/layout/region_shape.h"

namespace tachyon::zk {

// Allocated rows within a circuit.
using CircuitAllocations = absl::flat_hash_map<RegionColumn, Allocations>;

// - |start| is the current start row of the region (not of this column).
// - |slack| is the maximum number of rows the start could be moved down,
//   taking into account prior columns.
TACHYON_EXPORT std::optional<RowIndex> FirstFitRegion(
    CircuitAllocations* column_allocations,
    const std::vector<RegionColumn>& region_columns, RowIndex region_length,
    RowIndex start, std::optional<RowIndex> slack);

template <typename F>
struct RegionInfo {
  RegionInfo(RowIndex region_start, RegionShape<F>&& region)
      : region_start(region_start), region(std::move(region)) {}

  RowIndex region_start;
  RegionShape<F> region;
};

template <typename F>
struct SlotInResult {
  std::vector<RegionInfo<F>> regions;
  CircuitAllocations column_allocations;
};

// Positions the regions starting at the earliest row for which none of the
// columns are in use, taking into account gaps between earlier regions.
template <typename F>
SlotInResult<F> SlotIn(std::vector<RegionShape<F>>& region_shapes) {
  // Tracks the empty regions for each column.
  CircuitAllocations column_allocations;

  std::vector<RegionInfo<F>> regions;
  regions.reserve(region_shapes.size());

  for (RegionShape<F>& region : region_shapes) {
    // Sort the region's columns to ensure determinism.
    // NOTE(TomTaehoonKim): Sorted result might be different from the original
    // See
    // https://github.com/kroma-network/halo2/blob/7d0a36990452c8e7ebd600de258420781a9b7917/halo2_proofs/src/circuit/floor_planner/v1/strategy.rs#L171-L191
    // - An unstable sort is fine, because region.columns() returns a set.
    // - The sort order relies on Column's Ord implementation!
    std::vector<RegionColumn> region_columns(region.columns().begin(),
                                             region.columns().end());
    std::sort(region_columns.begin(), region_columns.end(),
              [](const RegionColumn& lhs, const RegionColumn& rhs) {
                return lhs < rhs;
              });

    std::optional<RowIndex> region_start =
        FirstFitRegion(&column_allocations, region_columns, region.row_count(),
                       0, std::nullopt);
    CHECK(region_start.has_value()) << "We can always fit a region somewhere";

    regions.emplace_back(*region_start, std::move(region));
  }

  return {regions, column_allocations};
}

TACHYON_EXPORT struct SlotInBiggestAdviceFirstResult {
  std::vector<RowIndex> region_starts;
  CircuitAllocations column_allocations;
};

// Sorts the regions by advice area and then lays them out with the |SlotIn|
// strategy.
template <typename F>
SlotInBiggestAdviceFirstResult SlotInBiggestAdviceFirst(
    const std::vector<RegionShape<F>>& region_shapes) {
  std::vector<RegionShape<F>> sorted_regions = region_shapes;
  // NOTE(TomTaehoonKim): Sorted result might be different from the original
  // See
  // https://github.com/kroma-network/halo2/blob/7d0a36990452c8e7ebd600de258420781a9b7917/halo2_proofs/src/layout/floor_planner/v1/strategy.rs#L202-L215
  // TODO(TomTaehoonKim): Implement pdqsort to make it same as the original
  // See
  // https://github.com/orlp/pdqsort
  std::sort(sorted_regions.begin(), sorted_regions.end(),
            [](const RegionShape<F>& lhs, const RegionShape<F>& rhs) {
              // Count the number of advice columns
              size_t lhs_advice_cols = 0;
              for (const RegionColumn& column : lhs.columns()) {
                if (column.type() == RegionColumn::Type::kColumn) {
                  const AnyColumnKey& c = column.column();
                  if (c.type() == ColumnType::kAdvice) {
                    ++lhs_advice_cols;
                  }
                }
              }
              size_t rhs_advice_cols = 0;
              for (const RegionColumn& column : rhs.columns()) {
                if (column.type() == RegionColumn::Type::kColumn) {
                  const AnyColumnKey& c = column.column();
                  if (c.type() == ColumnType::kAdvice) {
                    ++rhs_advice_cols;
                  }
                }
              }
              // Sort by advice area (since this has the most contention).
              return lhs_advice_cols * lhs.row_count() <
                     rhs_advice_cols * rhs.row_count();
            });
  std::reverse(sorted_regions.begin(), sorted_regions.end());

  // Lay out the sorted regions.
  SlotInResult<F> result = SlotIn(sorted_regions);

  // Un-sort the regions so they match the original indexing.
  std::sort(result.regions.begin(), result.regions.end(),
            [](const RegionInfo<F>& lhs, const RegionInfo<F>& rhs) {
              return lhs.region.region_index() < rhs.region.region_index();
            });
  std::vector<RowIndex> region_starts = base::Map(
      result.regions,
      [](const RegionInfo<F>& region) { return region.region_start; });

  return {region_starts, result.column_allocations};
}

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_LAYOUT_FLOOR_PLANNER_V1_V1_STRATEGY_H_
