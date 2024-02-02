// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_PERMUTATION_UNPERMUTED_TABLE_H_
#define TACHYON_ZK_PLONK_PERMUTATION_UNPERMUTED_TABLE_H_

#include <utility>
#include <vector>

#include "gtest/gtest_prod.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/range.h"
#include "tachyon/base/ref.h"
#include "tachyon/zk/plonk/permutation/label.h"
#include "tachyon/zk/plonk/permutation/permutation_utils.h"

namespace tachyon::zk::plonk {

// The |UnpermutedTable| contains elements that are the product-of-powers
// of δ and ω (called "label"). And each permutation polynomial (in evaluation
// form) is constructed by assigning elements in the |UnpermutedTable|.
//
// Let modulus = 2ˢ * T + 1, then
// |UnpermutedTable|
// = [[δⁱω⁰, δⁱω¹, δⁱω², ..., δⁱωⁿ⁻¹] for i in range(0..T-1)]
template <typename Evals>
class UnpermutedTable {
 public:
  using F = typename Evals::Field;
  using Table = std::vector<Evals>;

  UnpermutedTable() = default;

  const Table& table() const& { return table_; }

  const F& operator[](const Label& label) const {
    return *table_[label.col][label.row];
  }

  base::Ref<const Evals> GetColumn(size_t i) const {
    return base::Ref<const Evals>(&table_[i]);
  }

  std::vector<base::Ref<const Evals>> GetColumns(
      base::Range<size_t> range) const {
    CHECK_EQ(range.Intersect(base::Range<size_t>::Until(table_.size())), range);

    std::vector<base::Ref<const Evals>> ret;
    ret.reserve(range.GetSize());
    for (size_t i : range) {
      ret.push_back(GetColumn(i));
    }
    return ret;
  }

  template <typename Domain>
  static UnpermutedTable Construct(size_t cols, RowIndex rows,
                                   const Domain* domain) {
    // The ω is gᵀ with order 2ˢ where modulus = 2ˢ * T + 1.
    std::vector<F> omega_powers =
        domain->GetRootsOfUnity(rows, domain->group_gen());

    // The δ is g^2ˢ with order T where modulus = 2ˢ * T + 1.
    F delta = GetDelta<F>();

    Table unpermuted_table;
    unpermuted_table.reserve(cols);
    // Assign [δ⁰ω⁰, δ⁰ω¹, δ⁰ω², ..., δ⁰ωⁿ⁻¹] to the first col.
    unpermuted_table.push_back(Evals(std::move(omega_powers)));

    // Assign [δⁱω⁰, δⁱω¹, δⁱω², ..., δⁱωⁿ⁻¹] to each col.
    for (size_t i = 1; i < cols; ++i) {
      std::vector<F> col = base::CreateVector(rows, F::Zero());
      // TODO(dongchangYoo): Optimize this with
      // https://github.com/kroma-network/tachyon/pull/115.
      for (RowIndex j = 0; j < rows; ++j) {
        col[j] = *unpermuted_table[i - 1][j] * delta;
      }
      unpermuted_table.push_back(Evals(std::move(col)));
    }
    return UnpermutedTable(std::move(unpermuted_table));
  }

 private:
  explicit UnpermutedTable(Table table) : table_(std::move(table)) {}

  Table table_;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_UNPERMUTED_TABLE_H_
