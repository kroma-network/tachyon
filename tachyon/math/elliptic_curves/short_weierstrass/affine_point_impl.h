#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_IMPL_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_IMPL_H_

#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"

namespace tachyon {
namespace math {

#define CLASS AffinePoint<Curve, std::enable_if_t<Curve::kIsSWCurve>>

template <typename Curve>
constexpr ProjectivePoint<Curve> CLASS::DoubleProjective() const {
  if (IsZero()) {
    return ProjectivePoint<Curve>::Zero();
  }

  // https://hyperelliptic.org/EFD/g1p/auto-shortw-projective.html#doubling-mdbl-2007-bl
  // XX = X1²
  BaseField xx = x_;
  xx.SquareInPlace();

  // w = a + 3 * XX
  BaseField w = xx;
  w += w.Double();
  w += Curve::A();

  // Y1Y1 = Y1²
  BaseField y1y1 = y_;
  y1y1.SquareInPlace();

  // R = 2 * Y1Y1
  BaseField r = std::move(y1y1);
  r.DoubleInPlace();

  // sss = 4 * Y1 * R
  BaseField sss = y_;
  sss *= r;
  sss.DoubleInPlace().DoubleInPlace();

  // RR = R²
  BaseField rr = r;
  r.SquareInPlace();

  // B = (X1 + R)² - XX - RR
  BaseField b = std::move(r);
  b += x_;
  b.SquareInPlace();
  b -= xx;
  b -= rr;

  // h = w² - 2 * B
  BaseField h = w;
  h.SquareInPlace();
  h -= b.Double();

  // X3 = 2 * h * Y1
  BaseField x = std::move(y_);
  x *= h;
  x.DoubleInPlace();

  // Y3 = w * (B - h) - 2 * RR
  BaseField y = std::move(b);
  y -= h;
  y *= w;
  y -= rr.Double();

  // Z3 = sss
  BaseField z = std::move(sss);

  return {std::move(x), std::move(y), std::move(z)};
}

template <typename Curve>
constexpr PointXYZZ<Curve> CLASS::DoubleXYZZ() const {
  if (IsZero()) {
    return PointXYZZ<Curve>::Zero();
  }

  // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#doubling-mdbl-2008-s-1
  // U = 2 * Y1
  BaseField u = y_;
  u.DoubleInPlace();

  // V = U²
  BaseField v = u;
  v.SquareInPlace();

  // W = U * V
  BaseField w = u;
  w *= v;

  // S = X1 * V
  BaseField s = x_;
  s *= v;

  // M = 3 * X1² + a
  BaseField m = x_;
  m.SquareInPlace();
  m += m.Double();
  m += Curve::A();

  // X3 = M² - 2 * S
  BaseField x = m;
  x.SquareInPlace();
  x -= s.Double();

  // Y3 = M * (S - X3) - W * Y1
  std::array<BaseField, 2> a;
  a[0] = std::move(m);
  a[1] = w;
  a[1].NegInPlace();
  std::array<BaseField, 2> b;
  b[0] = std::move(s);
  b[0] -= x;
  b[1] = y_;
  BaseField y = BaseField::SumOfProducts(a, b);

  // ZZ3 = V
  BaseField zz = std::move(v);

  // ZZZ3 = W
  BaseField zzz = std::move(w);

  return {std::move(x), std::move(y), std::move(zz), std::move(zzz)};
}

#undef CLASS

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_IMPL_H_
