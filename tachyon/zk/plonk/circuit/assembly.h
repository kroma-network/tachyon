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
  Assembly(uint32_t k, std::vector<RationalEvals> fixeds,
           PermutationAssembly<PCSTy> permutation,
           std::vector<std::vector<bool>> selectors,
           base::Range<size_t> usable_rows)
      : k_(k),
        fixeds_(std::move(fixeds)),
        permutation_(std::move(permutation)),
        selectors_(std::move(selectors)),
        usable_rows_(usable_rows) {}

  uint32_t k() const { return k_; }
  const std::vector<RationalEvals>& fixeds() const { return fixeds_; }
  const PermutationAssembly<PCSTy>& permutation() const { return permutation_; }
  const std::vector<std::vector<bool>>& selectors() const { return selectors_; }
  const base::Range<size_t>& usable_rows() const { return usable_rows_; }

  // Assignment methods
  Error EnableSelector(std::string_view name, const Selector& selector,
                       size_t row) override {
    if (!usable_rows_.Contains(row)) {
      return Error::kNotEnoughRowsAvailable;
    }
    selectors_[selector.index()][row] = true;
    return Error::kNone;
  }

  Error QueryInstance(const InstanceColumnKey& column, size_t row,
                      Value<F>* instance) override {
    if (!usable_rows_.Contains(row)) {
      return Error::kNotEnoughRowsAvailable;
    }
    *instance = Value<F>::Unknown();
    return Error::kNone;
  }

  Error AssignFixed(std::string_view name, const FixedColumnKey& column,
                    size_t row, AssignCallback assign) override {
    if (!usable_rows_.Contains(row)) {
      return Error::kNotEnoughRowsAvailable;
    }
    fixeds_[column.index()] = std::move(assign).Run();
    return Error::kNone;
  }

  Error Copy(const AnyColumnKey& left_column, size_t left_row,
             const AnyColumnKey& right_column, size_t right_row) override {
    if (!(usable_rows_.Contains(left_row) &&
          usable_rows_.Contains(right_row))) {
      return Error::kNotEnoughRowsAvailable;
    }
    return permutation_.Copy(left_column, left_row, right_column, right_row);
  }

  Error FillFromRow(const FixedColumnKey& column, size_t from_row,
                    AssignCallback assign) override {
    if (!usable_rows_.Contains(from_row)) {
      return Error::kNotEnoughRowsAvailable;
    }
    math::RationalField<F> value = std::move(assign).Run();
    base::Range<size_t> range(usable_rows_.from + from_row, usable_rows_.to);
    for (size_t i : range) {
      std::ignore = i;
      fixeds_[column.index()] = value;
    }
    return Error::kNone;
  }

 private:
  uint32_t k_ = 0;
  std::vector<RationalEvals> fixeds_;
  PermutationAssembly<PCSTy> permutation_;
  std::vector<std::vector<bool>> selectors_;
  // A range of available rows for assignment and copies.
  base::Range<size_t> usable_rows_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_ASSEMBLY_H_
