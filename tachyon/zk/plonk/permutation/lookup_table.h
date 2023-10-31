// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_PERMUTATION_LOOKUP_TABLE_H_
#define TACHYON_ZK_PLONK_PERMUTATION_LOOKUP_TABLE_H_

#include <utility>
#include <vector>

#include "gtest/gtest_prod.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain.h"
#include "tachyon/zk/plonk/permutation/label.h"

namespace tachyon::zk {

// The |LookupTable| contains elements that are the product-of-powers
// of 𝛿 and w (called "label"). And each permutation polynomial (in evaluation
// form) is constructed by assigning elements in the |LookupTable|.
//
// Let modulus = 2ˢ * T + 1, then
// |LookupTable|
// = [[𝛿ⁱw⁰, 𝛿ⁱw¹, 𝛿ⁱw², ..., 𝛿ⁱwⁿ⁻¹] for i in range(0..T-1)]
template <typename F, size_t N>
class LookupTable {
 public:
  using Rows = std::vector<F>;
  using Table = std::vector<Rows>;

  LookupTable() = default;

  const F& operator[](const Label& label) const {
    return table_[label.col][label.row];
  }
  F& operator[](const Label& label) { return table_[label.col][label.row]; }

  static LookupTable Construct(
      size_t size, const math::UnivariateEvaluationDomain<F, N>* domain) {
    // The w is gᵀ with order 2ˢ where modulus = 2ˢ * T + 1.
    std::vector<F> omega_powers =
        domain->GetRootsOfUnity(N, domain->group_gen());

    // The 𝛿 is g^2ˢ with order T where modulus = 2ˢ * T + 1.
    F delta = GetDelta();

    Table lookup_table;
    lookup_table.reserve(size);
    // Assign [𝛿⁰w⁰, 𝛿⁰w¹, 𝛿⁰w², ..., 𝛿⁰wⁿ⁻¹] to the first row.
    lookup_table.push_back(std::move(omega_powers));

    // Assign [𝛿ⁱw⁰, 𝛿ⁱw¹, 𝛿ⁱw², ..., 𝛿ⁱwⁿ⁻¹] to each row.
    for (size_t i = 1; i < size; ++i) {
      Rows rows = base::CreateVector(N, F::Zero());
      // TODO(dongchangYoo): Optimize this with
      // https://github.com/kroma-network/tachyon/pull/115.
      for (size_t j = 0; j < N; ++j) {
        rows[j] = lookup_table[i - 1][j] * delta;
      }
      lookup_table.push_back(std::move(rows));
    }
    return LookupTable(std::move(lookup_table));
  }

 private:
  FRIEND_TEST(LookupTableTest, Construct);

  explicit LookupTable(Table table) : table_(std::move(table)) {}

  // Calculate 𝛿 = g^2ˢ with order T (i.e., T-th root of unity),
  // where T = F::Config::kTrace.
  constexpr static F GetDelta() {
    F g = F::FromMontgomery(F::Config::kSubgroupGenerator);
    typename F::BigIntTy s(F::Config::kTwoAdicity);
    F adicity = F(2).Pow(s);
    return g.Pow(adicity.ToBigInt());
  }

  Table table_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_LOOKUP_TABLE_H_
