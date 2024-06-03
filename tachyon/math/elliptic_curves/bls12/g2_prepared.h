// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_ELLIPTIC_CURVES_BLS12_G2_PREPARED_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_BLS12_G2_PREPARED_H_

#include <utility>

#include "tachyon/base/optional.h"
#include "tachyon/math/base/big_int.h"
#include "tachyon/math/base/bit_iterator.h"
#include "tachyon/math/elliptic_curves/pairing/g2_prepared_base.h"
#include "tachyon/math/elliptic_curves/pairing/g2_projective.h"

namespace tachyon::math::bls12 {

template <typename BLS12CurveConfig>
class G2Prepared : public G2PreparedBase<BLS12CurveConfig> {
 public:
  using Config = BLS12CurveConfig;
  using G2Curve = typename Config::G2Curve;
  using Fp2 = typename G2Curve::BaseField;
  using Fp = typename Fp2::BaseField;
  using G2AffinePoint = typename G2Curve::AffinePoint;

  G2Prepared() = default;
  explicit G2Prepared(const EllCoeffs<Fp2>& ell_coeffs)
      : G2PreparedBase<BLS12CurveConfig>(ell_coeffs) {}
  explicit G2Prepared(EllCoeffs<Fp2>&& ell_coeffs)
      : G2PreparedBase<BLS12CurveConfig>(std::move(ell_coeffs)) {}

  static G2Prepared From(const G2AffinePoint& q) {
    if (q.IsZero()) {
      return {};
    } else {
      Fp two_inv = unwrap<Fp>(Fp(2).Inverse());

      EllCoeffs<Fp2> ell_coeffs;
      size_t size = Config::kXLimbNums * 64;
      // NOTE(chokobole): A bit array consists of elements from [0, 1].
      // We reserve space in |ell_coeffs| assuming that these elements are
      // uniformly distributed.
      ell_coeffs.reserve(/*double=*/size + /*add=*/size / 2);

      G2Projective<Config> r(q.x(), q.y(), Fp2::One());
      auto it = BitIteratorBE<BigInt<Config::kXLimbNums>>::begin(
          &Config::kX,
          /*skip_leading_zeros=*/false);
      ++it;
      auto end = BitIteratorBE<BigInt<Config::kXLimbNums>>::end(&Config::kX);
      while (it != end) {
        ell_coeffs.push_back(r.DoubleInPlace(two_inv));
        if (*it) {
          ell_coeffs.push_back(r.AddInPlace(q));
        }
        ++it;
      }

      return G2Prepared(std::move(ell_coeffs));
    }
  }
};

}  // namespace tachyon::math::bls12

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BLS12_G2_PREPARED_H_
