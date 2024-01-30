// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_ARGUMENT_H_
#define TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_ARGUMENT_H_

#include <utility>
#include <vector>

#include "tachyon/base/containers/contains.h"
#include "tachyon/export.h"
#include "tachyon/zk/plonk/base/column_key.h"

namespace tachyon::zk {

// TODO(dongchangYoo): impl logics related to prove and verify.
class TACHYON_EXPORT PermutationArgument {
 public:
  PermutationArgument() = default;
  explicit PermutationArgument(const std::vector<AnyColumnKey>& columns)
      : columns_(columns) {}
  explicit PermutationArgument(std::vector<AnyColumnKey>&& columns)
      : columns_(std::move(columns)) {}

  const std::vector<AnyColumnKey>& columns() const { return columns_; }

  void AddColumn(const AnyColumnKey& column) {
    if (base::Contains(columns_, column)) return;
    columns_.push_back(column);
  }

  // Returns the minimum circuit degree required by the permutation argument.
  // The argument may use larger degree gates depending on the actual
  // circuit's degree and how many columns are involved in the permutation.
  size_t RequiredDegree() const {
    // degree 2:
    // l_first(X) * (1 - z(X)) = 0
    //
    // We will fit as many polynomials pᵢ(X) as possible
    // into the required degree of the circuit, so the
    // following will not affect the required degree of
    // this middleware.
    //
    // clang-format off
    // (1 - (l_last(X) + l_blind(X))) * (z(ω * X) Π (p(X) + β * sᵢ(X) + γ) - z(X) Π (p(X) + β * δⁱ * X + γ))
    // clang-format on
    //
    // On the first sets of columns, except the first
    // set, we will do
    //
    // l_first(X) * (z(X) - z'(ω^(last) X)) = 0
    //
    // where z'(X) is the permutation for the previous set
    // of columns.
    //
    // On the final set of columns, we will do
    //
    // degree 3:
    // l_last(X) * (z'(X)² - z'(X)) = 0
    //
    // which will allow the last value to be zero to
    // ensure the argument is perfectly complete.

    // There are constraints of degree 3 regardless of the
    // number of columns involved.
    return 3;
  }

 private:
  std::vector<AnyColumnKey> columns_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_ARGUMENT_H_
