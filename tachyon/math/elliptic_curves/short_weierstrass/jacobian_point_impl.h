#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_JACOBIAN_POINT_IMPL_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_JACOBIAN_POINT_IMPL_H_

#include <utility>

#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"

namespace tachyon::math {

#define CLASS    \
  JacobianPoint< \
      Curve, std::enable_if_t<Curve::kType == CurveType::kShortWeierstrass>>

template <typename Curve>
constexpr CLASS CLASS::Add(const JacobianPoint& other) const {
  if (IsZero()) {
    return other;
  }

  if (other.IsZero()) {
    return *this;
  }

  JacobianPoint ret;
  DoAdd(*this, other, ret);
  return ret;
}

template <typename Curve>
constexpr CLASS& CLASS::AddInPlace(const JacobianPoint& other) {
  if (IsZero()) {
    return *this = other;
  }

  if (other.IsZero()) {
    return *this;
  }

  DoAdd(*this, other, *this);
  return *this;
}

// static
template <typename Curve>
constexpr void CLASS::DoAdd(const JacobianPoint& a, const JacobianPoint& b,
                            JacobianPoint& c) {
  // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
  // Z1Z1 = Z1²
  BaseField z1z1 = a.z_.Square();

  // Z2Z2 = Z2²
  BaseField z2z2 = b.z_.Square();

  // U1 = X1 * Z2Z2
  BaseField u1 = a.x_ * z2z2;

  // U2 = X2 * Z1Z1
  BaseField u2 = b.x_ * z1z1;

  // S1 = Y1 * Z2 * Z2Z2
  BaseField s1 = a.y_ * b.z_;
  s1 *= z2z2;

  // S2 = Y2 * Z1 * Z1Z1
  BaseField s2 = b.y_ * a.z_;
  s2 *= z1z1;

  if (u1 == u2 && s1 == s2) {
    // The two points are equal, so we Double.
    if (&a == &c) {
      c.DoubleInPlace();
    } else {
      c = a.Double();
    }
  } else {
    // If we're adding -a and a together, c.z_ becomes zero as H becomes zero.

    // H = U2 - U1
    BaseField h = u2 - u1;

    // I = (2 * H)²
    BaseField i = h.Double();
    i.SquareInPlace();

    // J = -H * I
    BaseField j = h * i;
    j.NegInPlace();

    // r = 2 * (S2 - S1)
    BaseField r = s2 - s1;
    r.DoubleInPlace();

    // V = U1 * I
    BaseField v = u1 * i;

    // X3 = r² + J - 2 * V
    c.x_ = r.Square();
    c.x_ += j;
    c.x_ -= v.Double();

    // Y3 = r * (V - X3) + 2 * S1 * J
    BaseField lefts[] = {std::move(r), s1.Double()};
    BaseField rights[] = {v - c.x_, std::move(j)};
    c.y_ = BaseField::SumOfProductsSerial(lefts, rights);

    // Z3 = ((Z1 + Z2)² - Z1Z1 - Z2Z2) * H
    // This is equal to Z3 = 2 * Z1 * Z2 * H, and computing it this way is
    // faster.
    c.z_ = a.z_ * b.z_;
    c.z_.DoubleInPlace();
    c.z_ *= h;
  }
}

template <typename Curve>
constexpr CLASS CLASS::Add(const AffinePoint<Curve>& other) const {
  if (other.infinity()) return *this;
  if (IsZero()) {
    return JacobianPoint::FromAffine(other);
  }

  JacobianPoint ret;
  DoAdd(*this, other, ret);
  return ret;
}

template <typename Curve>
constexpr CLASS& CLASS::AddInPlace(const AffinePoint<Curve>& other) {
  if (other.infinity()) return *this;
  if (IsZero()) {
    return *this = JacobianPoint::FromAffine(other);
  }

  DoAdd(*this, other, *this);
  return *this;
}

// static
template <typename Curve>
constexpr void CLASS::DoAdd(const JacobianPoint& a, const AffinePoint<Curve>& b,
                            JacobianPoint& c) {
  // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
  // Z1Z1 = Z1²
  BaseField z1z1 = a.z_.Square();

  // U2 = X2 * Z1Z1
  BaseField u2 = b.x() * z1z1;

  // S2 = Y2 * Z1 * Z1Z1
  BaseField s2 = b.y() * a.z_;
  s2 *= z1z1;

  if (a.x_ == u2 && a.y_ == s2) {
    // The two points are equal, so we Double.
    if (&a == &c) {
      c.DoubleInPlace();
    } else {
      c = a.Double();
    }
  } else {
    // If we're adding -a and a together, c.z_ becomes zero as H becomes zero.

    // H = U2 - X1
    BaseField h = u2 - a.x_;

    // I = 4 * H²
    BaseField i = h.Square();
    i.DoubleInPlace().DoubleInPlace();

    // J = -H * I
    BaseField j = h * i;
    j.NegInPlace();

    // r = 2 * (S2 - Y1)
    BaseField r = s2 - a.y_;
    r.DoubleInPlace();

    // V = X1 * I
    BaseField v = a.x_ * i;

    // X3 = r² + J - 2 * V
    c.x_ = r.Square();
    c.x_ += j;
    c.x_ -= v.Double();

    // Y3 = r * (V - X3) + 2 * Y1 * J
    BaseField lefts[] = {std::move(r), a.y_.Double()};
    BaseField rights[] = {v - c.x_, std::move(j)};
    c.y_ = BaseField::SumOfProductsSerial(lefts, rights);

    // Z3 = 2 * Z1 * H;
    // Can alternatively be computed as (Z1 + H)² - Z1Z1 - HH, but the latter is
    // slower.
    c.z_ = a.z_ * h;
    c.z_.DoubleInPlace();
  }
}

template <typename Curve>
constexpr CLASS CLASS::DoDouble() const {
  if (IsZero()) {
    return JacobianPoint::Zero();
  }

  JacobianPoint ret;
  DoDoubleImpl(*this, ret);
  return ret;
}

template <typename Curve>
constexpr CLASS& CLASS::DoDoubleInPlace() {
  if (IsZero()) {
    return *this;
  }

  DoDoubleImpl(*this, *this);
  return *this;
}

// static
template <typename Curve>
constexpr void CLASS::DoDoubleImpl(const JacobianPoint& a, JacobianPoint& b) {
  if constexpr (Curve::Config::kAIsZero) {
    // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
    // XX = X1²
    BaseField xx = a.x_.Square();

    // YY = Y1²
    BaseField yy = a.y_.Square();

    // YYYY = YY²
    BaseField yyyy = yy.Square();

    // D = 2 * ((X1 + B)² - XX - YYYY)
    //   = 2 * ((X1 + Y1²)² - XX - YYYY)
    //   = 2 * 2 * X1 * Y1²
    BaseField d;
    if constexpr (BaseField::ExtensionDegree() == 1 ||
                  BaseField::ExtensionDegree() == 2) {
      d = a.x_ * yy;
      d.DoubleInPlace().DoubleInPlace();
    } else {
      d = a.x_ + yy;
      d.SquareInPlace();
      d -= xx;
      d -= yyyy;
      d.DoubleInPlace();
    }

    // E = 3 * XX
    BaseField e = xx.Double();
    e += xx;

    // Z3 = 2 * Y1 * Z1
    b.z_ = a.y_ * a.z_;
    b.z_.DoubleInPlace();

    // X3 = E² - 2 * D
    b.x_ = e.Square();
    b.x_ -= d.Double();

    // Y3 = E * (D - X3) - 8 * YYYY
    b.y_ = d - b.x_;
    b.y_ *= e;
    b.y_ -= yyyy.DoubleInPlace().DoubleInPlace().DoubleInPlace();
  } else {
    // https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian.html#doubling-dbl-2007-bl
    // XX = X1²
    BaseField xx = a.x_.Square();

    // YY = Y1²
    BaseField yy = a.y_.Square();

    // YYYY = YY²
    BaseField yyyy = yy.Square();

    // ZZ = Z1²
    BaseField zz = a.z_.Square();

    // S = 2 * ((X1 + YY)² - XX - YYYY)
    BaseField s = a.x_ + yy;
    s.SquareInPlace();
    s -= xx;
    s -= yyyy;
    s.DoubleInPlace();

    // M = 3 * XX + a * ZZ²
    BaseField m = xx.Double();
    m += xx;
    if constexpr (!Curve::Config::kAIsZero) {
      m += Curve::Config::MulByA(zz.Square());
    }

    // T = M² - 2 * S
    // X3 = T
    b.x_ = m.Square();
    b.x_ -= s.Double();

    // Z3 = (Y1 + Z1)² - YY - ZZ
    // Can be calculated as Z3 = 2 * Y1 * Z1, and this is faster.
    b.z_ = a.y_ + a.z_;
    b.z_.DoubleInPlace();

    // Y3 = M * (S - X3) - 8 * YYYY
    b.y_ = s - b.x_;
    b.y_ *= m;
    b.y_ -= yyyy.DoubleInPlace().DoubleInPlace().DoubleInPlace();
  }
}

#undef CLASS

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_JACOBIAN_POINT_IMPL_H_
