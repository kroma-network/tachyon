// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_COMMITTED_H_
#define TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_COMMITTED_H_

#include <utility>
#include <vector>

#include "tachyon/zk/base/blinded_polynomial.h"

namespace tachyon::zk::plonk {

// Committed polynomials for chunks of columns.
template <typename Poly>
class PermutationCommitted {
 public:
  PermutationCommitted() = default;
  explicit PermutationCommitted(
      std::vector<BlindedPolynomial<Poly>>&& product_polys)
      : product_polys_(std::move(product_polys)) {}

  const std::vector<BlindedPolynomial<Poly>>& product_polys() const {
    return product_polys_;
  }

  std::vector<BlindedPolynomial<Poly>>&& TakeProductPolys() && {
    return std::move(product_polys_);
  }

 private:
  std::vector<BlindedPolynomial<Poly>> product_polys_;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_COMMITTED_H_
