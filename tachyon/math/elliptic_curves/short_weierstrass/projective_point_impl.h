#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_PROJECTIVE_POINT_IMPL_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_PROJECTIVE_POINT_IMPL_H_

#include <utility>

#include "tachyon/math/elliptic_curves/short_weierstrass/projective_point.h"
namespace tachyon::math {

#define CLASS      \
  ProjectivePoint< \
      Curve, std::enable_if_t<Curve::kType == CurveType::kShortWeierstrass>>

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
  BaseField y1z2 = y_ * other.z_;

  // X1Z2 = X1 * Z2
  BaseField x1z2 = x_ * other.z_;

  // Z1Z2 = Z1 * Z2
  BaseField z1z2 = z_ * other.z_;

  // u = Y2 * Z1 - Y1Z2
  BaseField u = other.y_ * z_;
  u -= y1z2;

  // v = X2 * Z1 - X1Z2
  BaseField v = other.x_ * z_;
  v -= x1z2;

  if (u.IsZero() && v.IsZero()) {
    return DoubleInPlace();
  }

  // uu = u²
  BaseField uu = u.Square();

  // vv = v²
  BaseField vv = v.Square();

  // vvv = v * vv
  BaseField vvv = v * vv;

  // R = vv * X1Z2
  BaseField r = vv * x1z2;

  // A = uu * Z1Z2 - vvv - 2 * R
  BaseField a = uu * z1z2;
  a -= vvv;
  a -= r.Double();

  // X3 = v * A
  x_ = v * a;

  // Y3 = u * (R - A) - vvv * Y1Z2
  BaseField lefts[] = {std::move(u), -vvv};
  BaseField rights[] = {r - a, std::move(y1z2)};
  y_ = BaseField::SumOfProductsSerial(lefts, rights);

  // Z3 = vvv * Z1Z2
  z_ = vvv * z1z2;

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
  BaseField u = other.y() * z_;
  u -= y_;

  // v = X2 * Z1 - X1
  BaseField v = other.x() * z_;
  v -= x_;

  if (u.IsZero() && v.IsZero()) {
    return DoubleInPlace();
  }

  // uu = u²
  BaseField uu = u.Square();

  // vv = v²
  BaseField vv = v.Square();

  // vvv = v * vv
  BaseField vvv = v * vv;

  // R = vv * X1
  BaseField r = vv * x_;

  // A = uu * Z1 - vvv - 2 * R
  BaseField a = uu * z_;
  a -= vvv;
  a -= r.Double();

  // X3 = v * A
  x_ = v * a;

  // Y3 = u * (R - A) - vvv * Y1
  BaseField lefts[] = {std::move(u), -vvv};
  BaseField rights[] = {r - a, std::move(y_)};
  y_ = BaseField::SumOfProductsSerial(lefts, rights);

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
  BaseField xx = x_.Square();

  // ZZ = Z1²
  BaseField zz = z_.Square();

  // w = a * ZZ + 3 * XX
  BaseField w = xx.Double();
  w += xx;
  if constexpr (!Curve::Config::kAIsZero) {
    w += Curve::Config::MulByA(zz);
  }

  // s = 2 * Y1 * Z1
  BaseField s = y_ * z_;
  s.DoubleInPlace();

  // ss = s²
  BaseField ss = s.Square();

  // sss = s * ss
  BaseField sss = s * ss;

  // R = Y1 * s
  BaseField r = y_ * s;

  // RR = R²
  BaseField rr = r.Square();

  // B = (X1 + R)² - XX - RR
  BaseField b = x_ + r;
  b.SquareInPlace();
  b -= xx;
  b -= rr;

  // h = w² - 2 * B
  BaseField h = w.Square();
  h -= b.Double();

  // X3 = h * s
  x_ = h * s;

  // Y3 = w * (B - h) - 2 * RR
  y_ = b - h;
  y_ *= w;
  y_ -= rr.Double();

  // Z3 = sss
  z_ = std::move(sss);

  return *this;
}

#undef CLASS

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_PROJECTIVE_POINT_IMPL_H_
