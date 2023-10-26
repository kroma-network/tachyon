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
#include "tachyon/zk/plonk/circuit/region.h"
#include "tachyon/zk/plonk/circuit/region_column.h"

namespace tachyon::zk {

template <typename F>
class RegionShape : public Region<F>::Layouter {
 public:
  using AnnotateCallback = typename Region<F>::Layouter::AnnotateCallback;
  using AssignCallback = typename Region<F>::Layouter::AssignCallback;

  RegionShape() = default;
  explicit RegionShape(size_t region_index) : region_index_(region_index) {}

  size_t region_index() const { return region_index_; }
  const absl::flat_hash_set<RegionColumn>& columns() const { return columns_; }
  size_t row_count() const { return row_count_; }

  // Layouter methods
  Error EnableSelector(AnnotateCallback annotate, const Selector& selector,
                       size_t offset) override {
    columns_.insert(RegionColumn(selector));
    row_count_ = std::max(row_count_, offset + 1);
    return Error::kNone;
  }

  Error AssignAdvice(AnnotateCallback annotate, const AdviceColumn& column,
                     size_t offset, AssignCallback to, Cell* cell) override {
    columns_.insert(RegionColumn(column));
    row_count_ = std::max(row_count_, offset + 1);
    *cell = Cell(region_index_, offset, column);
    return Error::kNone;
  }

  Error AssignAdviceFromConstant(AnnotateCallback annotate,
                                 const AdviceColumn& column, size_t offset,
                                 math::RationalField<F> constant,
                                 Cell* cell) override {
    return AssignAdvice(
        std::move(annotate), column, offset,
        [constant = std::move(constant)]() {
          return Value<F>::Known(std::move(constant));
        },
        cell);
  }

  Error AssignAdviceFromInstance(AnnotateCallback annotate,
                                 const InstanceColumn& instance, size_t row,
                                 const InstanceColumn& advice, size_t offset,
                                 AssignedCell<F>* cell) override {
    columns_.insert(RegionColumn(advice));
    row_count_ = std::max(row_count_, offset + 1);
    *cell = Cell(region_index_, offset, advice);
    return Error::kNone;
  }

  Error AssignFixed(AnnotateCallback annotate, const FixedColumn& column,
                    size_t offset, AssignCallback to, Cell* cell) override {
    columns_.insert(RegionColumn(column));
    row_count_ = std::max(row_count_, offset + 1);
    *cell = Cell(region_index_, offset, column);
    return Error::kNone;
  }

 private:
  size_t region_index_ = 0;
  absl::flat_hash_set<RegionColumn> columns_;
  size_t row_count_ = 0;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_REGION_SHAPE_H_
