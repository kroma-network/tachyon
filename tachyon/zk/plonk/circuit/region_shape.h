// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_REGION_SHAPE_H_
#define TACHYON_ZK_PLONK_CIRCUIT_REGION_SHAPE_H_

#include <algorithm>
#include <utility>

#include "absl/container/flat_hash_set.h"

#include "tachyon/export.h"
#include "tachyon/zk/plonk/circuit/region_column.h"
#include "tachyon/zk/plonk/circuit/region_layouter.h"

namespace tachyon::zk {

template <typename F>
class RegionShape : public RegionLayouter<F> {
 public:
  using AssignCallback = typename RegionLayouter<F>::AssignCallback;

  explicit RegionShape(size_t region_index) : region_index_(region_index) {}

  size_t region_index() const { return region_index_; }
  const absl::flat_hash_set<RegionColumn>& columns() const { return columns_; }
  RowIndex row_count() const { return row_count_; }

  // RegionLayouter methods
  void EnableSelector(std::string_view, const Selector& selector,
                      RowIndex offset) override {
    UpdateColumnsAndRowCount(selector, offset);
  }

  Cell AssignAdvice(std::string_view, const AdviceColumnKey& column,
                    RowIndex offset, AssignCallback) override {
    UpdateColumnsAndRowCount(column, offset);
    return {region_index_, offset, column};
  }

  Cell AssignAdviceFromConstant(
      std::string_view name, const AdviceColumnKey& column, RowIndex offset,
      const math::RationalField<F>& constant) override {
    return AssignAdvice(name, column, offset, AssignCallback());
  }

  AssignedCell<F> AssignAdviceFromInstance(std::string_view,
                                           const InstanceColumnKey&, RowIndex,
                                           const AdviceColumnKey& advice,
                                           RowIndex offset) override {
    UpdateColumnsAndRowCount(advice, offset);
    Cell cell(region_index_, offset, advice);
    return {std::move(cell), Value<F>::Unknown()};
  }

  Cell AssignFixed(std::string_view, const FixedColumnKey& column,
                   RowIndex offset, AssignCallback) override {
    UpdateColumnsAndRowCount(column, offset);
    return {region_index_, offset, column};
  }

 private:
  template <typename T>
  void UpdateColumnsAndRowCount(const T& arg, RowIndex offset) {
    columns_.insert(RegionColumn(arg));
    row_count_ = std::max(row_count_, offset + 1);
  }

  size_t region_index_ = 0;
  absl::flat_hash_set<RegionColumn> columns_;
  RowIndex row_count_ = 0;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_REGION_SHAPE_H_
