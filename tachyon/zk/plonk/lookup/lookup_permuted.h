// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_LOOKUP_LOOKUP_PERMUTED_H_
#define TACHYON_ZK_PLONK_LOOKUP_LOOKUP_PERMUTED_H_

#include <utility>

#include "tachyon/zk/base/blinded_polynomial.h"
#include "tachyon/zk/base/evals_pair.h"

namespace tachyon::zk {

template <typename Poly, typename Evals>
class LookupPermuted {
 public:
  using F = typename Poly::Field;

  LookupPermuted() = default;
  LookupPermuted(EvalsPair<Evals> compressed_evals_pair,
                 EvalsPair<Evals> permuted_evals_pair,
                 BlindedPolynomial<Poly> permuted_input_poly,
                 BlindedPolynomial<Poly> permuted_table_poly)
      : compressed_evals_pair_(std::move(compressed_evals_pair)),
        permuted_evals_pair_(std::move(permuted_evals_pair)),
        permuted_input_poly_(std::move(permuted_input_poly)),
        permuted_table_poly_(std::move(permuted_table_poly)) {}

  const EvalsPair<Evals>& compressed_evals_pair() const {
    return compressed_evals_pair_;
  }
  const EvalsPair<Evals>& permuted_evals_pair() const {
    return permuted_evals_pair_;
  }
  BlindedPolynomial<Poly>&& permuted_input_poly() && {
    return std::move(permuted_input_poly_);
  }
  BlindedPolynomial<Poly>&& permuted_table_poly() && {
    return std::move(permuted_table_poly_);
  }

 private:
  EvalsPair<Evals> compressed_evals_pair_;
  EvalsPair<Evals> permuted_evals_pair_;
  BlindedPolynomial<Poly> permuted_input_poly_;
  BlindedPolynomial<Poly> permuted_table_poly_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_LOOKUP_LOOKUP_PERMUTED_H_
