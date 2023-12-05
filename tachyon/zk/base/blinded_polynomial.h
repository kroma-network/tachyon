// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_BASE_BLINDED_POLYNOMIAL_H_
#define TACHYON_ZK_BASE_BLINDED_POLYNOMIAL_H_

#include <utility>

#include "tachyon/base/ref.h"

namespace tachyon::zk {

template <typename T>
class Ref;

template <typename Poly>
class BlindedPolynomial {
 public:
  using F = typename Poly::Field;

  BlindedPolynomial() = default;
  BlindedPolynomial(Poly&& poly, const F& blind)
      : poly_(std::move(poly)), blind_(blind) {}
  BlindedPolynomial(Poly&& poly, F&& blind)
      : poly_(std::move(poly)), blind_(std::move(blind)) {}

  const Poly& poly() const { return poly_; }
  const F& blind() const { return blind_; }

  base::Ref<const BlindedPolynomial<Poly>> ToRef() const {
    return base::Ref<const BlindedPolynomial<Poly>>(this);
  }

 private:
  Poly poly_;
  F blind_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_BLINDED_POLYNOMIAL_H_
