#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_PROJECTIVE_POINT_IMPL_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_PROJECTIVE_POINT_IMPL_H_

#include "tachyon/math/elliptic_curves/short_weierstrass/projective_point.h"

namespace tachyon {
namespace math {

#define CLASS ProjectivePoint<Curve, std::enable_if_t<Curve::kIsSWCurve>>

template <typename Curve>
constexpr CLASS& CLASS::AddInPlace(const ProjectivePoint& other) {
  if (IsZero()) {
    return *this = other;
  }

  if (other.IsZero()) {
    return *this;
  }

  // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-projective.html#addition-add-1998-cmo-2
  // Y1Z2 = Y1 * Z2
  BaseField y1z2 = y_;
  y1z2 *= other.z_;

  // X1Z2 = X1 * Z2
  BaseField x1z2 = x_;
  x1z2 *= other.z_;

  // Z1Z2 = Z1 * Z2
  BaseField z1z2 = z_;
  z1z2 *= other.z_;

  // u = Y2 * Z1 - Y1Z2
  BaseField u = other.y_;
  u *= z_;
  u -= y1z2;

  // v = X2 * Z1 - X1Z2
  BaseField v = other.x_;
  v *= z_;
  v -= x1z2;

  if (u.IsZero() && v.IsZero()) {
    return DoubleInPlace();
  }

  // uu = u²
  BaseField uu = u;
  uu.SquareInPlace();

  // vv = v²
  BaseField vv = v;
  vv.SquareInPlace();

  // vvv = v * vv
  BaseField vvv = v;
  vvv *= vv;

  // R = vv * X1Z2
  BaseField r = std::move(vv);
  r *= x1z2;

  // A = uu * Z1Z2 - vvv - 2 * R
  BaseField a = std::move(uu);
  a *= z1z2;
  a -= vvv;
  a -= r.Double();

  // X3 = v * A
  x_ = std::move(v);
  x_ *= a;

  // Y3 = u * (R - A) - vvv * Y1Z2
  std::array<BaseField, 2> lefts;
  lefts[0] = std::move(u);
  lefts[1] = vvv;
  lefts[1].NegInPlace();
  std::array<BaseField, 2> rights;
  rights[0] = std::move(r);
  rights[0] -= a;
  rights[1] = std::move(y1z2);
  y_ = BaseField::SumOfProducts(lefts, rights);

  // Z3 = vvv * Z1Z2
  z_ = std::move(vvv);
  z_ *= z1z2;

  return *this;
}

template <typename Curve>
constexpr CLASS& CLASS::AddInPlace(const AffinePoint<Curve>& other) {
  if (IsZero()) {
    return *this = other.ToProjective();
  }

  if (other.IsZero()) {
    return *this;
  }

  // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-projective.html#addition-madd-1998-cmo
  // u = Y2 * Z1 - Y1
  BaseField u = other.y();
  u *= z_;
  u -= y_;

  // v = X2 * Z1 - X1
  BaseField v = other.x();
  v *= z_;
  v -= x_;

  if (u.IsZero() && v.IsZero()) {
    return DoubleInPlace();
  }

  // uu = u²
  BaseField uu = u;
  uu.SquareInPlace();

  // vv = v²
  BaseField vv = v;
  vv.SquareInPlace();

  // vvv = v * vv
  BaseField vvv = v;
  vvv *= vv;

  // R = vv * X1
  BaseField r = std::move(vv);
  r *= x_;

  // A = uu * Z1 - vvv - 2 * R
  BaseField a = std::move(uu);
  a *= z_;
  a -= vvv;
  a -= r.Double();

  // X3 = v * A
  x_ = std::move(v);
  x_ *= a;

  // Y3 = u * (R - A) - vvv * Y1
  std::array<BaseField, 2> lefts;
  lefts[0] = std::move(u);
  lefts[1] = vvv;
  lefts[1].NegInPlace();
  std::array<BaseField, 2> rights;
  rights[0] = std::move(r);
  rights[0] -= a;
  rights[1] = std::move(y_);
  y_ = BaseField::SumOfProducts(lefts, rights);

  // Z3 = vvv * Z1
  z_ *= vvv;

  return *this;
}

template <typename Curve>
constexpr CLASS& CLASS::DoubleInPlace() {
  if (IsZero()) {
    return *this;
  }

  // https://hyperelliptic.org/EFD/g1p/auto-shortw-projective.html#doubling-dbl-2007-bl
  // XX = X1²
  BaseField xx = x_;
  xx.SquareInPlace();

  // ZZ = Z1²
  BaseField zz = z_;
  zz.SquareInPlace();

  // w = a * ZZ + 3 * XX
  BaseField w = xx;
  w += w.Double();
  w += Curve::A() * zz;

  // s = 2 * Y1 * Z1
  BaseField s = y_;
  s *= z_;
  s.DoubleInPlace();

  // ss = s²
  BaseField ss = s;
  ss.SquareInPlace();

  // sss = s * ss
  BaseField sss = s;
  sss *= ss;

  // R = Y1 * s
  BaseField r = y_;
  r *= s;

  // RR = R²
  BaseField rr = r;
  rr.SquareInPlace();

  // B = (X1 + R)² - XX - RR
  BaseField b = x_;
  b += r;
  b.SquareInPlace();
  b -= xx;
  b -= rr;

  // h = w² - 2 * B
  BaseField h = w;
  h.SquareInPlace();
  h -= b.Double();

  // X3 = h * s
  x_ = std::move(s);
  x_ *= h;

  // Y3 = w * (B - h) - 2 * RR
  y_ = std::move(b);
  y_ -= h;
  y_ *= w;
  y_ -= rr.Double();

  // Z3 = sss
  z_ = std::move(sss);

  return *this;
}

#undef CLASS

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_PROJECTIVE_POINT_IMPL_H_
