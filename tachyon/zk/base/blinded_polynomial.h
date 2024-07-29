// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_BASE_BLINDED_POLYNOMIAL_H_
#define TACHYON_ZK_BASE_BLINDED_POLYNOMIAL_H_

#include <string>
#include <utility>

#include "absl/strings/substitute.h"

namespace tachyon::zk {

template <typename Poly, typename Evals>
class BlindedPolynomial {
 public:
  using F = typename Poly::Field;

  BlindedPolynomial() = default;
  BlindedPolynomial(Evals&& evals, const F& blind)
      : evals_(std::move(evals)), blind_(blind) {}
  BlindedPolynomial(Evals&& evals, F&& blind)
      : evals_(std::move(evals)), blind_(std::move(blind)) {}
  BlindedPolynomial(Poly&& poly, const F& blind)
      : poly_(std::move(poly)), blind_(blind) {}
  BlindedPolynomial(Poly&& poly, F&& blind)
      : poly_(std::move(poly)), blind_(std::move(blind)) {}

  const Poly& poly() const { return poly_; }
  const Evals& evals() const { return evals_; }
  const F& blind() const { return blind_; }

  Poly&& TakePoly() { return std::move(poly_); }
  Evals&& TakeEvals() { return std::move(evals_); }

  void set_poly(Poly&& poly) { poly_ = std::move(poly); }
  void set_evals(Evals&& evals) { evals_ = std::move(evals); }

  template <typename Domain>
  void TransformEvalsToPoly(const Domain* domain) {
    poly_ = domain->IFFT(std::move(evals_));
  }

  std::string ToString() const {
    if (evals_.NumElements() == 0) {
      return absl::Substitute("{poly: $0, blind: $1}", poly_.ToString(),
                              blind_.ToString());
    } else {
      return absl::Substitute("{evals: $0, blind: $1}", evals_.ToString(),
                              blind_.ToString());
    }
  }

 private:
  Poly poly_;
  Evals evals_;
  F blind_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_BLINDED_POLYNOMIAL_H_
