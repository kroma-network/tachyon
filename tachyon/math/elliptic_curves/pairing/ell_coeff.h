// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_ELLIPTIC_CURVES_PAIRING_ELL_COEFF_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_PAIRING_ELL_COEFF_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"

namespace tachyon::math {

template <typename F>
class EllCoeff {
 public:
  EllCoeff() = default;
  EllCoeff(const F& c0, const F& c1, const F& c2) : c0_(c0), c1_(c1), c2_(c2) {}
  EllCoeff(F&& c0, F&& c1, F&& c2)
      : c0_(std::move(c0)), c1_(std::move(c1)), c2_(std::move(c2)) {}

  const F& c0() const { return c0_; }
  const F& c1() const { return c1_; }
  const F& c2() const { return c2_; }

  std::string ToString() const {
    return absl::Substitute("{c0: $0, c1: $1, c2: $2}", c0_.ToString(),
                            c1_.ToString(), c2_.ToString());
  }

 private:
  F c0_;
  F c1_;
  F c2_;
};

template <typename F>
using EllCoeffs = std::vector<EllCoeff<F>>;

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_PAIRING_ELL_COEFF_H_
