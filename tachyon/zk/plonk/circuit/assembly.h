// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_ASSEMBLY_H_
#define TACHYON_ZK_PLONK_CIRCUIT_ASSEMBLY_H_

#include <tuple>
#include <utility>
#include <vector>

#include "tachyon/base/logging.h"
#include "tachyon/base/range.h"
#include "tachyon/math/base/rational_field.h"
#include "tachyon/zk/plonk/circuit/assignment.h"
#include "tachyon/zk/plonk/permutation/permutation_assembly.h"

namespace tachyon::zk {

template <typename PCSTy>
class Assembly : public Assignment<typename PCSTy::Field> {
 public:
  using F = typename PCSTy::Field;
  using RationalEvals = typename PCSTy::RationalEvals;
  using AssignCallback = typename Assignment<F>::AssignCallback;

  Assembly() = default;
  Assembly(uint32_t k, std::vector<RationalEvals>&& fixed_columns,
           PermutationAssembly<PCSTy>&& permutation,
           std::vector<std::vector<bool>>&& selectors,
           base::Range<size_t> usable_rows)
      : k_(k),
        fixed_columns_(std::move(fixed_columns)),
        permutation_(std::move(permutation)),
        selectors_(std::move(selectors)),
        usable_rows_(usable_rows) {}

  uint32_t k() const { return k_; }
  const std::vector<RationalEvals>& fixed_columns() const {
    return fixed_columns_;
  }
  const PermutationAssembly<PCSTy>& permutation() const { return permutation_; }
  const std::vector<std::vector<bool>>& selectors() const { return selectors_; }
  const base::Range<size_t>& usable_rows() const { return usable_rows_; }

  // Assignment methods
  void EnableSelector(std::string_view name, const Selector& selector,
                      size_t row) override {
    CHECK(usable_rows_.Contains(row)) << "Not enough rows available";
    selectors_[selector.index()][row] = true;
  }

  Value<F> QueryInstance(const InstanceColumnKey& column, size_t row) override {
    CHECK(usable_rows_.Contains(row)) << "Not enough rows available";
    return Value<F>::Unknown();
  }

  void AssignFixed(std::string_view name, const FixedColumnKey& column,
                   size_t row, AssignCallback assign) override {
    CHECK(usable_rows_.Contains(row)) << "Not enough rows available";
    fixed_columns_[column.index()] = std::move(assign).Run();
  }

  void Copy(const AnyColumnKey& left_column, size_t left_row,
            const AnyColumnKey& right_column, size_t right_row) override {
    CHECK(usable_rows_.Contains(left_row) && usable_rows_.Contains(right_row))
        << "Not enough rows available";
    permutation_.Copy(left_column, left_row, right_column, right_row);
  }

  void FillFromRow(const FixedColumnKey& column, size_t from_row,
                   AssignCallback assign) override {
    CHECK(usable_rows_.Contains(from_row)) << "Not enough rows available";
    math::RationalField<F> value = std::move(assign).Run();
    base::Range<size_t> range(usable_rows_.from + from_row, usable_rows_.to);
    for (size_t i : range) {
      std::ignore = i;
      fixed_columns_[column.index()] = value;
    }
  }

 private:
  uint32_t k_ = 0;
  std::vector<RationalEvals> fixed_columns_;
  PermutationAssembly<PCSTy> permutation_;
  std::vector<std::vector<bool>> selectors_;
  // A range of available rows for assignment and copies.
  base::Range<size_t> usable_rows_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_ASSEMBLY_H_
