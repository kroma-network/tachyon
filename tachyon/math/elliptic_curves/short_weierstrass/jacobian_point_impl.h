#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_JACOBIAN_POINT_IMPL_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_JACOBIAN_POINT_IMPL_H_

#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"

namespace tachyon {
namespace math {

#define CLASS JacobianPoint<Curve, std::enable_if_t<Curve::kIsSWCurve>>

template <typename Curve>
constexpr CLASS& CLASS::AddInPlace(const JacobianPoint& other) {
  if (IsZero()) {
    return *this = other;
  }

  if (other.IsZero()) {
    return *this;
  }

  // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
  // Z1Z1 = Z1²
  BaseField z1z1 = z_.Square();

  // Z2Z2 = Z2²
  BaseField z2z2 = other.z_.Square();

  // U1 = X1 * Z2Z2
  BaseField u1 = x_;
  u1 *= z2z2;

  // U2 = X2 * Z1Z1
  BaseField u2 = other.x_;
  u2 *= z1z1;

  // S1 = Y1 * Z2 * Z2Z2
  BaseField s1 = y_;
  s1 *= other.z_;
  s1 *= z2z2;

  // S2 = Y2 * Z1 * Z1Z1
  BaseField s2 = other.y_;
  s2 *= z_;
  s2 *= z1z1;

  if (u1 == u2 && s1 == s2) {
    // The two points are equal, so we Double.
    DoubleInPlace();
  } else {
    // If we're adding -a and a together, z_ becomes zero as H becomes zero.

    // H = U2 - U1
    BaseField h = u2;
    h -= u1;

    // I = (2 * H)²
    BaseField i = h;
    i.DoubleInPlace().SquareInPlace();

    // J = -H * I
    BaseField j = h;
    j.NegInPlace();
    j *= i;

    // r = 2 * (S2 - S1)
    BaseField r = std::move(s2);
    r -= s1;
    r.DoubleInPlace();

    // V = U1 * I
    BaseField v = std::move(u1);
    v *= i;

    // X3 = r² + J - 2 * V
    x_ = r;
    x_.SquareInPlace();
    x_ += j;
    x_ -= v.Double();

    // Y3 = r * (V - X3) + 2 * S1 * J
    v -= x_;
    y_ = s1;
    y_.DoubleInPlace();
    std::array<BaseField, 2> a;
    a[0] = std::move(r);
    a[1] = y_;
    std::array<BaseField, 2> b;
    b[0] = std::move(v);
    b[1] = std::move(j);
    y_ = BaseField::SumOfProducts(a, b);

    // Z3 = ((Z1 + Z2)² - Z1Z1 - Z2Z2) * H
    // This is equal to Z3 = 2 * Z1 * Z2 * H, and computing it this way is
    // faster.
    z_ *= other.z_;
    z_.DoubleInPlace();
    z_ *= h;
  }
  return *this;
}

template <typename Curve>
constexpr CLASS& CLASS::AddInPlace(const AffinePoint<Curve>& other) {
  if (other.infinity()) return *this;
  if (IsZero()) {
    return *this = JacobianPoint::FromAffine(other);
  }

  // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
  // Z1Z1 = Z1²
  BaseField z1z1 = z_;
  z1z1.SquareInPlace();

  // U2 = X2 * Z1Z1
  BaseField u2 = other.x();
  u2 *= z1z1;

  // S2 = Y2 * Z1 * Z1Z1
  BaseField s2 = other.y();
  s2 *= z_;
  s2 *= z1z1;

  if (x_ == u2 && y_ == s2) {
    // The two points are equal, so we double.
    DoubleInPlace();
  } else {
    // If we're adding -a and a together, z_ becomes zero as H becomes zero.

    // H = U2 - X1
    BaseField h = u2;
    h -= x_;

    // HH = H²
    BaseField hh = h;
    hh.SquareInPlace();

    // I = 4 * HH
    BaseField i = hh;
    i.DoubleInPlace().DoubleInPlace();

    // J = -H * I
    BaseField j = h;
    j.NegInPlace();
    j *= i;

    // r = 2 * (S2 - Y1)
    BaseField r = s2;
    r -= y_;
    r.DoubleInPlace();

    // V = X1 * I
    BaseField v = x_;
    v *= i;

    // X3 = r² + J - 2 * V
    x_ = r.Square();
    x_ += j;
    x_ -= v.Double();

    // Y3 = r * (V - X3) + 2 * Y1 * J
    std::array<BaseField, 2> a;
    a[0] = std::move(r);
    a[1] = y_;
    a[1].DoubleInPlace();
    std::array<BaseField, 2> b;
    b[0] = std::move(v);
    b[0] -= x_;
    b[1] = std::move(j);
    y_ = BaseField::SumOfProducts(a, b);

    // Z3 = 2 * Z1 * H;
    // Can alternatively be computed as (Z1 + H)² - Z1Z1 - HH, but the latter is
    // slower.
    z_ *= h;
    z_.DoubleInPlace();
  }
  return *this;
}

template <typename Curve>
constexpr CLASS& CLASS::DoubleInPlace() {
  if (IsZero()) {
    return *this;
  }

  if (Curve::A().IsZero()) {
    // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
    // A = X1²
    BaseField a = x_;
    a.SquareInPlace();

    // B = Y1²
    BaseField b = y_;
    b.SquareInPlace();

    // C = B²
    BaseField c = b;
    c.SquareInPlace();

    // D = 2 * ((X1 + B)² - A - C)
    //   = 2 * ((X1 + Y1²)² - A - C)
    //   = 2 * 2 * X1 * Y1²
    uint64_t ext_deg = BaseField::Config::ExtensionDegree();
    BaseField d;
    if (ext_deg == 1 || ext_deg == 2) {
      d = x_;
      d *= b;
      d.DoubleInPlace().DoubleInPlace();
    } else {
      d = x_;
      d += b;
      d.SquareInPlace();
      d -= a;
      d -= c;
      d.SquareInPlace();
    }

    // E = 3 * A
    BaseField e = a + a.Double();

    // Z3 = 2 * Y1 * Z1
    z_ *= y_;
    z_.DoubleInPlace();

    // F = E²
    // X3 = F - 2 * D
    x_ = e;
    x_.SquareInPlace();
    x_ -= d.Double();

    // Y3 = E * (D - X3) - 8 * C
    y_ = std::move(d);
    y_ -= x_;
    y_ *= e;
    y_ -= c.DoubleInPlace().DoubleInPlace().DoubleInPlace();
  } else {
    // https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian.html#doubling-dbl-2007-bl
    // XX = X1²
    BaseField xx = x_.Square();

    // YY = Y1²
    BaseField yy = y_.Square();

    // YYYY = YY²
    BaseField yyyy = yy;
    yyyy.SquareInPlace();

    // ZZ = Z1²
    BaseField zz = z_;
    zz.SquareInPlace();

    // S = 2 * ((X1 + YY)² - XX - YYYY)
    BaseField s = ((x_ + yy).Square() - xx - yyyy).Double();

    // M = 3 * XX + a * ZZ²
    BaseField m = xx;
    m.DoubleInPlace();
    m += xx;
    m += Curve::A() * zz.Square();

    // T = M² - 2 * S
    // X3 = T
    x_ = m;
    x_.SquareInPlace();
    x_ -= s.Double();

    // Z3 = (Y1 + Z1)² - YY - ZZ
    // Can be calculated as Z3 = 2 * Y1 * Z1, and this is faster.
    z_ *= y_;
    z_.DoubleInPlace();

    // Y3 = M * (S - X3) - 8 * YYYY
    y_ = std::move(s);
    y_ -= x_;
    y_ *= m;
    y_ -= yyyy.DoubleInPlace().DoubleInPlace().DoubleInPlace();
  }
  return *this;
}

#undef CLASS

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_JACOBIAN_POINT_IMPL_H_
