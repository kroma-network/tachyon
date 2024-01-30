// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_KEYS_ASSEMBLY_H_
#define TACHYON_ZK_PLONK_KEYS_ASSEMBLY_H_

#include <utility>
#include <vector>

#include "tachyon/base/logging.h"
#include "tachyon/base/range.h"
#include "tachyon/math/base/rational_field.h"
#include "tachyon/zk/plonk/layout/assignment.h"
#include "tachyon/zk/plonk/permutation/permutation_assembly.h"

namespace tachyon::zk {

template <typename PCS>
class Assembly : public Assignment<typename PCS::Field> {
 public:
  using F = typename PCS::Field;
  using RationalEvals = typename PCS::RationalEvals;
  using AssignCallback = typename Assignment<F>::AssignCallback;

  Assembly() = default;
  Assembly(std::vector<RationalEvals>&& fixed_columns,
           PermutationAssembly<PCS>&& permutation,
           std::vector<std::vector<bool>>&& selectors,
           base::Range<RowIndex> usable_rows)
      : fixed_columns_(std::move(fixed_columns)),
        permutation_(std::move(permutation)),
        selectors_(std::move(selectors)),
        usable_rows_(usable_rows) {}

  const std::vector<RationalEvals>& fixed_columns() const {
    return fixed_columns_;
  }
  const PermutationAssembly<PCS>& permutation() const { return permutation_; }
  const std::vector<std::vector<bool>>& selectors() const { return selectors_; }
  const base::Range<RowIndex>& usable_rows() const { return usable_rows_; }

  // Assignment methods
  void EnableSelector(std::string_view, const Selector& selector,
                      RowIndex row) override {
    CHECK(usable_rows_.Contains(row)) << "Not enough rows available";
    selectors_[selector.index()][row] = true;
  }

  Value<F> QueryInstance(const InstanceColumnKey& column,
                         RowIndex row) override {
    CHECK(usable_rows_.Contains(row)) << "Not enough rows available";
    return Value<F>::Unknown();
  }

  void AssignFixed(std::string_view, const FixedColumnKey& column, RowIndex row,
                   AssignCallback assign) override {
    CHECK(usable_rows_.Contains(row)) << "Not enough rows available";
    *fixed_columns_[column.index()][row] = std::move(assign).Run().value();
  }

  void Copy(const AnyColumnKey& left_column, RowIndex left_row,
            const AnyColumnKey& right_column, RowIndex right_row) override {
    CHECK(usable_rows_.Contains(left_row) && usable_rows_.Contains(right_row))
        << "Not enough rows available";
    permutation_.Copy(left_column, left_row, right_column, right_row);
  }

  void FillFromRow(const FixedColumnKey& column, RowIndex from_row,
                   const math::RationalField<F>& value) override {
    CHECK(usable_rows_.Contains(from_row)) << "Not enough rows available";
    base::Range<RowIndex> range(usable_rows_.from + from_row, usable_rows_.to);
    RationalEvals& fixed_column = fixed_columns_[column.index()];
    for (RowIndex row : range) {
      *fixed_column[row] = value;
    }
  }

 private:
  std::vector<RationalEvals> fixed_columns_;
  PermutationAssembly<PCS> permutation_;
  std::vector<std::vector<bool>> selectors_;
  // A range of available rows for assignment and copies.
  base::Range<RowIndex> usable_rows_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_KEYS_ASSEMBLY_H_
