// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_SQUARE_ROOT_ALGORITHMS_SHANKS_H_
#define TACHYON_MATH_FINITE_FIELDS_SQUARE_ROOT_ALGORITHMS_SHANKS_H_

#include <utility>

namespace tachyon::math {

template <typename F>
constexpr bool ComputeShanksSquareRoot(const F& a, F* ret) {
  // https://eprint.iacr.org/2012/685.pdf (page 9, algorithm 2)
  // clang-format off
  // a² = b
  // a⁴ = b²
  //    = b^(p+1) (since b^(p-1) = 1, See https://en.wikipedia.org/wiki/Fermat%27s_little_theorem)
  // a  = b^((p + 1) / 4)
  // clang-format on
  F sqrt = a.Pow(F::Config::kModulusPlusOneDivFour);
  if (sqrt.Square() == a) {
    *ret = std::move(sqrt);
    return true;
  }
  return false;
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_SQUARE_ROOT_ALGORITHMS_SHANKS_H_
