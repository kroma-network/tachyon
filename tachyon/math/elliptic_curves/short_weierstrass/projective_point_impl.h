#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_PROJECTIVE_POINT_IMPL_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_PROJECTIVE_POINT_IMPL_H_

#include <utility>

#include "tachyon/math/elliptic_curves/short_weierstrass/projective_point.h"
namespace tachyon::math {

#define CLASS      \
  ProjectivePoint< \
      Curve, std::enable_if_t<Curve::kType == CurveType::kShortWeierstrass>>

template <typename Curve>
constexpr CLASS CLASS::Add(const ProjectivePoint& other) const {
  if (IsZero()) {
    return other;
  }

  if (other.IsZero()) {
    return *this;
  }

  ProjectivePoint ret;
  DoAdd(*this, other, ret);
  return ret;
}

template <typename Curve>
constexpr CLASS& CLASS::AddInPlace(const ProjectivePoint& other) {
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
constexpr void CLASS::DoAdd(const ProjectivePoint& a, const ProjectivePoint& b,
                            ProjectivePoint& c) {
  // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-projective.html#addition-add-1998-cmo-2
  // Y1Z2 = Y1 * Z2
  BaseField y1z2 = a.y_ * b.z_;

  // X1Z2 = X1 * Z2
  BaseField x1z2 = a.x_ * b.z_;

  // Z1Z2 = Z1 * Z2
  BaseField z1z2 = a.z_ * b.z_;

  // u = Y2 * Z1 - Y1Z2
  BaseField u = b.y_ * a.z_;
  u -= y1z2;

  // v = X2 * Z1 - X1Z2
  BaseField v = b.x_ * a.z_;
  v -= x1z2;

  if (u.IsZero() && v.IsZero()) {
    if (&a == &c) {
      c.DoubleInPlace();
    } else {
      c = a.Double();
    }
    return;
  }

  // uu = u²
  BaseField uu = u.Square();

  // vv = v²
  BaseField vv = v.Square();

  // vvv = v * vv
  BaseField vvv = v * vv;

  // R = vv * X1Z2
  BaseField r = vv * x1z2;

  // D = uu * Z1Z2 - vvv - 2 * R
  BaseField d = uu * z1z2;
  d -= vvv;
  d -= r.Double();

  // X3 = v * D
  c.x_ = v * d;

  // Y3 = u * (R - D) - vvv * Y1Z2
  BaseField lefts[] = {std::move(u), -vvv};
  BaseField rights[] = {r - d, std::move(y1z2)};
  c.y_ = BaseField::SumOfProductsSerial(lefts, rights);

  // Z3 = vvv * Z1Z2
  c.z_ = vvv * z1z2;
}

template <typename Curve>
constexpr CLASS CLASS::Add(const AffinePoint<Curve>& other) const {
  if (IsZero()) {
    return other.ToProjective();
  }

  if (other.IsZero()) {
    return *this;
  }

  ProjectivePoint ret;
  DoAdd(*this, other, ret);
  return ret;
}

template <typename Curve>
constexpr CLASS& CLASS::AddInPlace(const AffinePoint<Curve>& other) {
  if (IsZero()) {
    return *this = other.ToProjective();
  }

  if (other.IsZero()) {
    return *this;
  }

  DoAdd(*this, other, *this);
  return *this;
}

// static
template <typename Curve>
constexpr void CLASS::DoAdd(const ProjectivePoint& a,
                            const AffinePoint<Curve>& b, ProjectivePoint& c) {
  // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-projective.html#addition-madd-1998-cmo
  // u = Y2 * Z1 - Y1
  BaseField u = b.y() * a.z_;
  u -= a.y_;

  // v = X2 * Z1 - X1
  BaseField v = b.x() * a.z_;
  v -= a.x_;

  if (u.IsZero() && v.IsZero()) {
    if (&a == &c) {
      c.DoubleInPlace();
    } else {
      c = a.Double();
    }
    return;
  }

  // uu = u²
  BaseField uu = u.Square();

  // vv = v²
  BaseField vv = v.Square();

  // vvv = v * vv
  BaseField vvv = v * vv;

  // R = vv * X1
  BaseField r = vv * a.x_;

  // D = uu * Z1 - vvv - 2 * R
  BaseField d = uu * a.z_;
  d -= vvv;
  d -= r.Double();

  // X3 = v * D
  c.x_ = v * d;

  // Y3 = u * (R - D) - vvv * Y1
  BaseField lefts[] = {std::move(u), -vvv};
  BaseField rights[] = {r - d, a.y_};
  c.y_ = BaseField::SumOfProductsSerial(lefts, rights);

  // Z3 = vvv * Z1
  c.z_ = vvv * a.z_;
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
