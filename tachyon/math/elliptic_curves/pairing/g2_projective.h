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
  using Fp2 = typename G2Curve::BaseField;
  using Fp = typename Fp2::BaseField;
  using G2AffinePoint = typename G2Curve::AffinePoint;

  G2Projective() = default;
  G2Projective(const Fp2& x, const Fp2& y, const Fp2& z)
      : x_(x), y_(y), z_(z) {}
  G2Projective(Fp2&& x, Fp2&& y, Fp2&& z)
      : x_(std::move(x)), y_(std::move(y)), z_(std::move(z)) {}

  const Fp2& x() const { return x_; }
  const Fp2& y() const { return y_; }
  const Fp2& z() const { return z_; }

  // TODO(chokobole): Leave a comment to help understand readers.
  EllCoeff<Fp2> AddInPlace(const G2AffinePoint& q) {
    // Formula for line function when working with
    // homogeneous projective coordinates.
    Fp2 theta = y_ - (q.y() * z_);
    Fp2 lambda = x_ - (q.x() * z_);
    Fp2 c = theta.Square();
    Fp2 d = lambda.Square();
    Fp2 e = lambda * d;
    Fp2 f = z_ * c;
    Fp2 g = x_ * d;
    Fp2 h = e + f - g.Double();
    x_ = lambda * h;
    y_ = theta * (g - h) - (e * y_);
    z_ *= e;
    Fp2 j = theta * q.x() - (lambda * q.y());

    if constexpr (Config::kTwistType == TwistType::kM) {
      return {j, -theta, lambda};
    } else {
      return {lambda, -theta, j};
    }
  }

  // TODO(chokobole): Leave a comment to help understand readers.
  EllCoeff<Fp2> DoubleInPlace(const Fp& two_inv) {
    // Formula for line function when working with
    // homogeneous projective coordinates.
    Fp2 a = x_ * y_;
    a *= two_inv;
    Fp2 b = y_.Square();
    Fp2 c = z_.Square();
    Fp2 e = G2Curve::Config::kB * (c.Double() + c);
    Fp2 f = e.Double() + e;
    Fp2 g = b + f;
    g *= two_inv;
    Fp2 h = (y_ + z_).Square() - (b + c);
    Fp2 i = e - b;
    Fp2 j = x_.Square();
    Fp2 e_square = e.Square();

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
  Fp2 x_;
  Fp2 y_;
  Fp2 z_;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_PAIRING_G2_PROJECTIVE_H_
