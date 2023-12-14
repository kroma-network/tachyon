// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_ELLIPTIC_CURVES_PAIRING_G2_PROJECTIVE_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_PAIRING_G2_PROJECTIVE_H_

#include <utility>

#include "tachyon/math/elliptic_curves/pairing/ell_coeff.h"
#include "tachyon/math/elliptic_curves/pairing/twist_type.h"

namespace tachyon::math {

template <typename PairingFriendlyCurveConfig>
class G2Projective {
 public:
  using Config = PairingFriendlyCurveConfig;
  using G2Curve = typename Config::G2Curve;
  using Fp2Ty = typename G2Curve::BaseField;
  using FpTy = typename Fp2Ty::BaseField;
  using G2AffinePointTy = typename G2Curve::AffinePointTy;

  G2Projective() = default;
  G2Projective(const Fp2Ty& x, const Fp2Ty& y, const Fp2Ty& z)
      : x_(x), y_(y), z_(z) {}
  G2Projective(Fp2Ty&& x, Fp2Ty&& y, Fp2Ty&& z)
      : x_(std::move(x)), y_(std::move(y)), z_(std::move(z)) {}

  const Fp2Ty& x() const { return x_; }
  const Fp2Ty& y() const { return y_; }
  const Fp2Ty& z() const { return z_; }

  // TODO(chokobole): Leave a comment to help understand readers.
  EllCoeff<Fp2Ty> AddInPlace(const G2AffinePointTy& q) {
    // Formula for line function when working with
    // homogeneous projective coordinates.
    Fp2Ty theta = y_ - (q.y() * z_);
    Fp2Ty lambda = x_ - (q.x() * z_);
    Fp2Ty c = theta.Square();
    Fp2Ty d = lambda.Square();
    Fp2Ty e = lambda * d;
    Fp2Ty f = z_ * c;
    Fp2Ty g = x_ * d;
    Fp2Ty h = e + f - g.Double();
    x_ = lambda * h;
    y_ = theta * (g - h) - (e * y_);
    z_ *= e;
    Fp2Ty j = theta * q.x() - (lambda * q.y());

    if constexpr (Config::kTwistType == TwistType::kM) {
      return {j, -theta, lambda};
    } else {
      return {lambda, -theta, j};
    }
  }

  // TODO(chokobole): Leave a comment to help understand readers.
  EllCoeff<Fp2Ty> DoubleInPlace(const FpTy& two_inv) {
    // Formula for line function when working with
    // homogeneous projective coordinates.
    Fp2Ty a = x_ * y_;
    a *= two_inv;
    Fp2Ty b = y_.Square();
    Fp2Ty c = z_.Square();
    Fp2Ty e = G2Curve::Config::kB * (c.Double() + c);
    Fp2Ty f = e.Double() + e;
    Fp2Ty g = b + f;
    g *= two_inv;
    Fp2Ty h = (y_ + z_).Square() - (b + c);
    Fp2Ty i = e - b;
    Fp2Ty j = x_.Square();
    Fp2Ty e_square = e.Square();

    x_ = a * (b - f);
    y_ = g.Square() - (e_square.Double() + e_square);
    z_ = b * h;

    if constexpr (Config::kTwistType == TwistType::kM) {
      return {i, j.Double() + j, -h};
    } else {
      return {-h, j.Double() + j, i};
    }
  }

  G2Projective& NegateInPlace() {
    y_ = -y_;
    return *this;
  }

 private:
  Fp2Ty x_;
  Fp2Ty y_;
  Fp2Ty z_;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_PAIRING_G2_PROJECTIVE_H_
