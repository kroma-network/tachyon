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
#include "tachyon/zk/base/ref.h"
#include "tachyon/zk/plonk/permutation/label.h"

namespace tachyon::zk {

// The |UnpermutedTable| contains elements that are the product-of-powers
// of 𝛿 and w (called "label"). And each permutation polynomial (in evaluation
// form) is constructed by assigning elements in the |UnpermutedTable|.
//
// Let modulus = 2ˢ * T + 1, then
// |UnpermutedTable|
// = [[𝛿ⁱw⁰, 𝛿ⁱw¹, 𝛿ⁱw², ..., 𝛿ⁱwⁿ⁻¹] for i in range(0..T-1)]
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

  Ref<const Evals> GetColumn(size_t i) const {
    return Ref<const Evals>(&table_[i]);
  }

  std::vector<Ref<const Evals>> GetColumns(base::Range<size_t> range) const {
    CHECK_EQ(range.Intersect(base::Range<size_t>::Until(table_.size())), range);

    std::vector<Ref<const Evals>> ret;
    ret.reserve(range.GetSize());
    for (size_t i : range) {
      ret.push_back(GetColumn(i));
    }
    return ret;
  }

  template <typename Domain>
  static UnpermutedTable Construct(size_t size, const Domain* domain) {
    constexpr static size_t kMaxDegree = Domain::kMaxDegree;

    // The w is gᵀ with order 2ˢ where modulus = 2ˢ * T + 1.
    std::vector<F> omega_powers =
        domain->GetRootsOfUnity(kMaxDegree + 1, domain->group_gen());

    // The 𝛿 is g^2ˢ with order T where modulus = 2ˢ * T + 1.
    F delta = GetDelta();

    Table unpermuted_table;
    unpermuted_table.reserve(size);
    // Assign [𝛿⁰w⁰, 𝛿⁰w¹, 𝛿⁰w², ..., 𝛿⁰wⁿ⁻¹] to the first col.
    unpermuted_table.push_back(Evals(std::move(omega_powers)));

    // Assign [𝛿ⁱw⁰, 𝛿ⁱw¹, 𝛿ⁱw², ..., 𝛿ⁱwⁿ⁻¹] to each col.
    for (size_t i = 1; i < size; ++i) {
      std::vector<F> col = base::CreateVector(kMaxDegree + 1, F::Zero());
      // TODO(dongchangYoo): Optimize this with
      // https://github.com/kroma-network/tachyon/pull/115.
      for (size_t j = 0; j <= kMaxDegree; ++j) {
        col[j] = *unpermuted_table[i - 1][j] * delta;
      }
      unpermuted_table.push_back(Evals(std::move(col)));
    }
    return UnpermutedTable(std::move(unpermuted_table));
  }

 private:
  FRIEND_TEST(UnpermutedTableTest, Construct);
  FRIEND_TEST(UnpermutedTableTest, GetColumns);

  explicit UnpermutedTable(Table table) : table_(std::move(table)) {}

  // Calculate 𝛿 = g^2ˢ with order T (i.e., T-th root of unity),
  // where T = F::Config::kTrace.
  constexpr static F GetDelta() {
    F g = F::FromMontgomery(F::Config::kSubgroupGenerator);
    F adicity = F(2).Pow(F::Config::kTwoAdicity);
    return g.Pow(adicity.ToBigInt());
  }

  Table table_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_UNPERMUTED_TABLE_H_
