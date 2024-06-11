// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_ELLIPTIC_CURVES_BN_G2_PREPARED_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_BN_G2_PREPARED_H_

#include <utility>

#include "tachyon/base/optional.h"
#include "tachyon/math/elliptic_curves/pairing/g2_prepared_base.h"
#include "tachyon/math/elliptic_curves/pairing/g2_projective.h"

namespace tachyon::math::bn {

template <typename BNCurveConfig>
class G2Prepared : public G2PreparedBase<BNCurveConfig> {
 public:
  using Config = BNCurveConfig;
  using G2Curve = typename Config::G2Curve;
  using Fp2 = typename G2Curve::BaseField;
  using Fp = typename Fp2::BaseField;
  using G2AffinePoint = typename G2Curve::AffinePoint;

  G2Prepared() = default;
  explicit G2Prepared(const EllCoeffs<Fp2>& ell_coeffs)
      : G2PreparedBase<BNCurveConfig>(ell_coeffs) {}
  explicit G2Prepared(EllCoeffs<Fp2>&& ell_coeffs)
      : G2PreparedBase<BNCurveConfig>(std::move(ell_coeffs)) {}

  static G2Prepared From(const G2AffinePoint& q) {
    if (q.IsZero()) {
      return {};
    } else {
      G2Projective<Config> r(q.x(), q.y(), Fp2::One());

      EllCoeffs<Fp2> ell_coeffs;
      // NOTE(chokobole): |Config::kAteLoopCount| consists of elements
      // from [-1, 0, 1]. We reserve space in |ell_coeffs| assuming that these
      // elements are uniformly distributed.
      size_t size = std::size(Config::kAteLoopCount);
      ell_coeffs.reserve(/*double=*/size + /*add=*/size * 2 / 3);

      G2AffinePoint neg_q = -q;

      Fp two_inv = unwrap(Fp(2).Inverse());
      // NOTE(chokobole): skip the fist.
      for (size_t i = size - 2; i != SIZE_MAX; --i) {
        ell_coeffs.push_back(r.DoubleInPlace(two_inv));

        switch (Config::kAteLoopCount[i]) {
          case 1:
            ell_coeffs.push_back(r.AddInPlace(q));
            break;
          case -1:
            ell_coeffs.push_back(r.AddInPlace(neg_q));
            break;
          default:
            continue;
        }
      }

      G2AffinePoint q1 = MulByCharacteristic(q);
      G2AffinePoint q2 = MulByCharacteristic(q1);
      q2.NegateInPlace();

      if constexpr (Config::kXIsNegative) {
        r.NegateInPlace();
      }

      ell_coeffs.push_back(r.AddInPlace(q1));
      ell_coeffs.push_back(r.AddInPlace(q2));

      return G2Prepared(std::move(ell_coeffs));
    }
  }

 private:
  static G2AffinePoint MulByCharacteristic(const G2AffinePoint& r) {
    Fp2 x = r.x();
    x.FrobeniusMapInPlace(1);
    x *= Config::kTwistMulByQX;
    Fp2 y = r.y();
    y.FrobeniusMapInPlace(1);
    y *= Config::kTwistMulByQY;
    return G2AffinePoint(std::move(x), std::move(y));
  }
};

}  // namespace tachyon::math::bn

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BN_G2_PREPARED_H_
