// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_PLAN_REGION_H_
#define TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_PLAN_REGION_H_

#include <stddef.h>

#include <utility>
#include <vector>

#include "tachyon/math/base/rational_field.h"
#include "tachyon/zk/plonk/base/column_key.h"
#include "tachyon/zk/plonk/circuit/assignment.h"
#include "tachyon/zk/plonk/circuit/floor_planner/constant.h"
#include "tachyon/zk/plonk/circuit/region_layouter.h"

namespace tachyon::zk {

template <typename F>
class PlanRegion : public RegionLayouter<F> {
 public:
  using AssignCallback = typename RegionLayouter<F>::AssignCallback;

  PlanRegion(Assignment<F>* assignment, const std::vector<RowIndex>& regions,
             size_t region_index, Constants<F>& constant)
      : assignment_(assignment),
        regions_(regions),
        region_index_(region_index),
        constants_(constant) {}

  const Constants<F>& constants() const { return constants_; }

  // RegionLayouter methods
  void EnableSelector(std::string_view name, const Selector& selector,
                      RowIndex offset) override {
    assignment_->EnableSelector(name, selector,
                                regions_[region_index_] + offset);
  }

  void NameColumn(std::string_view name, const AnyColumnKey& column) override {
    assignment_->NameColumn(name, column);
  }

  Cell AssignAdvice(std::string_view name, const AdviceColumnKey& column,
                    RowIndex offset, AssignCallback assign) override {
    assignment_->AssignAdvice(name, column, regions_[region_index_] + offset,
                              std::move(assign));
    return {region_index_, offset, column};
  }

  Cell AssignAdviceFromConstant(
      std::string_view name, const AdviceColumnKey& column, RowIndex offset,
      const math::RationalField<F>& constant) override {
    Cell cell = AssignAdvice(name, column, offset, [&constant]() {
      return Value<math::RationalField<F>>::Known(constant);
    });
    ConstrainConstant(cell, constant);
    return cell;
  }

  AssignedCell<F> AssignAdviceFromInstance(std::string_view name,
                                           const InstanceColumnKey& instance,
                                           RowIndex row,
                                           const AdviceColumnKey& advice,
                                           RowIndex offset) override {
    Value<F> value = assignment_->QueryInstance(instance, row);

    Cell cell = AssignAdvice(name, advice, offset, [&value]() {
      return Value<math::RationalField<F>>::Known(
          math::RationalField<F>(value.value()));
    });

    assignment_->Copy(cell.column(),
                      regions_[cell.region_index()] + cell.row_offset(),
                      instance, row);

    return {std::move(cell), std::move(value)};
  }

  Cell AssignFixed(std::string_view name, const FixedColumnKey& column,
                   RowIndex offset, AssignCallback assign) override {
    assignment_->AssignFixed(name, column, regions_[region_index_] + offset,
                             std::move(assign));
    return {region_index_, offset, column};
  }

  void ConstrainConstant(const Cell& cell,
                         const math::RationalField<F>& constant) override {
    constants_.emplace_back(constant, cell);
  }

  void ConstrainEqual(const Cell& left, const Cell& right) override {
    assignment_->Copy(
        left.column(), regions_[left.region_index()] + left.row_offset(),
        right.column(), regions_[right.region_index()] + right.row_offset());
  }

 private:
  // not owned
  Assignment<F>* const assignment_;
  const std::vector<RowIndex>& regions_;
  const size_t region_index_;
  Constants<F>& constants_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_PLAN_REGION_H_
