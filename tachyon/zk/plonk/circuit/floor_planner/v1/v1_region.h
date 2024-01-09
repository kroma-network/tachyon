// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_V1_V1_REGION_H_
#define TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_V1_V1_REGION_H_

#include <stddef.h>

#include <utility>

#include "tachyon/zk/plonk/circuit/floor_planner/v1/v1_plan.h"
#include "tachyon/zk/plonk/circuit/region.h"

namespace tachyon::zk {

template <typename F>
class V1Region : public Region<F>::Layouter {
 public:
  using AssignCallback = typename Region<F>::Layouter::AssignCallback;

  V1Region(V1Plan<F>* plan, size_t region_index)
      : plan_(plan), region_index_(region_index) {}

  // Region<F>::Layouter methods
  void EnableSelector(std::string_view name, const Selector& selector,
                      size_t offset) override {
    plan_->assignment()->EnableSelector(
        name, selector, plan_->regions()[region_index_] + offset);
  }

  Cell AssignAdvice(std::string_view name, const AdviceColumnKey& column,
                    size_t offset, AssignCallback assign) override {
    plan_->assignment()->AssignAdvice(name, column,
                                      plan_->regions()[region_index_] + offset,
                                      std::move(assign));
    return {region_index_, offset, column};
  }

  void NameColumn(std::string_view name, const AnyColumnKey& column) override {
    plan_->assignment()->NameColumn(name, column);
  }

  Cell AssignAdviceFromConstant(
      std::string_view name, const AdviceColumnKey& column, size_t offset,
      const math::RationalField<F>& constant) override {
    Cell cell = AssignAdvice(name, column, offset, [&constant]() {
      return Value<math::RationalField<F>>::Known(constant);
    });
    ConstrainConstant(cell, constant);
    return cell;
  }

  AssignedCell<F> AssignAdviceFromInstance(std::string_view name,
                                           const InstanceColumnKey& instance,
                                           size_t row,
                                           const AdviceColumnKey& advice,
                                           size_t offset) override {
    Value<F> value = plan_->assignment()->QueryInstance(instance, row);

    Cell cell = AssignAdvice(name, advice, offset, [&value]() {
      return Value<math::RationalField<F>>::Known(
          math::RationalField<F>(value.value()));
    });

    plan_->assignment()->Copy(
        cell.column(),
        plan_->regions()[cell.region_index()] + cell.row_offset(), instance,
        row);

    return {std::move(cell), std::move(value)};
  }

  Cell AssignFixed(std::string_view name, const FixedColumnKey& column,
                   size_t offset, AssignCallback assign) override {
    plan_->assignment()->AssignFixed(name, column,
                                     plan_->regions()[region_index_] + offset,
                                     std::move(assign));
    return {region_index_, offset, column};
  }

  void ConstrainConstant(const Cell& cell,
                         const math::RationalField<F>& constant) override {
    plan_->constants().emplace_back(constant, cell);
  }

  void ConstrainEqual(const Cell& left, const Cell& right) override {
    plan_->assignment()->Copy(
        left.column(),
        plan_->regions()[left.region_index()] + left.row_offset(),
        right.column(),
        plan_->regions()[right.region_index()] + right.row_offset());
  }

 private:
  // not owned
  V1Plan<F>* const plan_;
  size_t region_index_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_V1_V1_REGION_H_
