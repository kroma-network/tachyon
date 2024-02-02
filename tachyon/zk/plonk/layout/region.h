// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_LAYOUT_REGION_H_
#define TACHYON_ZK_PLONK_LAYOUT_REGION_H_

#include <memory>
#include <string>
#include <utility>

#include "tachyon/base/functional/callback.h"
#include "tachyon/zk/plonk/base/column_key.h"
#include "tachyon/zk/plonk/constraint_system/selector.h"
#include "tachyon/zk/plonk/layout/assigned_cell.h"
#include "tachyon/zk/plonk/layout/region_layouter.h"

namespace tachyon::zk::plonk {

template <typename F>
class Region {
 public:
  using AssignCallback = base::OnceCallback<Value<F>()>;

  explicit Region(RegionLayouter<F>* layouter) : layouter_(layouter) {}

  // See the comment above.
  void EnableSelector(std::string_view name, const Selector& selector,
                      RowIndex offset) {
    layouter_->EnableSelector(name, selector, offset);
  }

  // See the comment above.
  void NameColumn(std::string_view name, const AnyColumnKey& column) {
    layouter_->NameColumn(name, column);
  }

  // See the comment above.
  AssignedCell<F> AssignAdvice(std::string_view name,
                               const AdviceColumnKey& column, RowIndex offset,
                               AssignCallback assign) {
    Value<F> value = Value<F>::Unknown();
    Cell cell =
        layouter_->AssignAdvice(name, column, offset, [&value, &assign]() {
          value = std::move(assign).Run();
          return value.ToRationalFieldValue();
        });
    return {std::move(cell), std::move(value)};
  }

  // See the comment above.
  AssignedCell<F> AssignAdviceFromConstant(std::string_view name,
                                           const AdviceColumnKey& column,
                                           RowIndex offset, const F& constant) {
    Cell cell = layouter_->AssignAdviceFromConstant(
        name, column, offset, math::RationalField<F>(constant));
    return {std::move(cell), Value<F>::Known(constant)};
  }

  // See the comment above.
  AssignedCell<F> AssignAdviceFromInstance(std::string_view name,
                                           const InstanceColumnKey& instance,
                                           RowIndex row,
                                           const AdviceColumnKey& advice,
                                           RowIndex offset) {
    return layouter_->AssignAdviceFromInstance(name, instance, row, advice,
                                               offset);
  }

  // See the comment above.
  AssignedCell<F> AssignFixed(std::string_view name,
                              const FixedColumnKey& column, RowIndex offset,
                              AssignCallback assign) {
    Value<F> value = Value<F>::Unknown();
    Cell cell =
        layouter_->AssignFixed(name, column, offset, [&value, &assign]() {
          value = std::move(assign).Run();
          return value.ToRationalFieldValue();
        });
    return {std::move(cell), std::move(value)};
  }

  // See the comment above.
  void ConstrainConstant(const Cell& cell,
                         const math::RationalField<F>& constant) {
    layouter_->ConstrainConstant(cell, constant);
  }

  // See the comment above.
  void ConstrainEqual(const Cell& left, const Cell& right) {
    layouter_->ConstrainEqual(left, right);
  }

 private:
  // not owned
  RegionLayouter<F>* const layouter_;
};

template <typename F>
void Selector::Enable(Region<F>& region, RowIndex offset) const {
  region.EnableSelector("", *this, offset);
}

template <typename F>
AssignedCell<F> AssignedCell<F>::CopyAdvice(std::string_view name,
                                            Region<F>& region,
                                            const AdviceColumnKey& column,
                                            RowIndex offset) const {
  AssignedCell<F> ret =
      region.AssignAdvice(name, column, offset, [this]() { return value_; });
  region.ConstrainEqual(ret.cell_, cell_);
  return ret;
}

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_LAYOUT_REGION_H_
