#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_IMPL_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_IMPL_H_

#include <utility>

#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"

namespace tachyon::math {

#define CLASS        \
  AffinePoint<Curve, \
              std::enable_if_t<Curve::kType == CurveType::kShortWeierstrass>>

template <typename Curve>
constexpr ProjectivePoint<Curve> CLASS::DoubleProjective() const {
  if (IsZero()) {
    return ProjectivePoint<Curve>::Zero();
  }

  // https://hyperelliptic.org/EFD/g1p/auto-shortw-projective.html#doubling-mdbl-2007-bl
  // XX = X1²
  BaseField xx = x_.Square();

  // w = a + 3 * XX
  BaseField w = xx;
  w += w.Double();
  if constexpr (!Curve::Config::kAIsZero) {
    // TODO(chokobole): Implement constexpr version of Curve::Config::AddByA()
    // for GPU.
    w += Curve::Config::kA;
  }

  // R = 2 * Y1²
  BaseField r = y_.Square();
  r.DoubleInPlace();

  // sss = 4 * Y1 * R
  BaseField sss = y_ * r;
  sss.DoubleInPlace().DoubleInPlace();

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

  // X3 = 2 * h * Y1
  BaseField x = h * y_;
  x.DoubleInPlace();

  // Y3 = w * (B - h) - 2 * RR
  BaseField y = b - h;
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
  BaseField u = y_.Double();

  // V = U²
  BaseField v = u.Square();

  // W = U * V
  BaseField w = u * v;

  // S = X1 * V
  BaseField s = x_ * v;

  // M = 3 * X1² + a
  BaseField m = x_.Square();
  m += m.Double();
  if constexpr (!Curve::Config::kAIsZero) {
    // TODO(chokobole): Implement constexpr version of Curve::Config::AddByA()
    // for GPU.
    m += Curve::Config::kA;
  }

  // X3 = M² - 2 * S
  BaseField x = m.Square();
  x -= s.Double();

  // Y3 = M * (S - X3) - W * Y1
  BaseField lefts[] = {std::move(m), -w};
  BaseField rights[] = {s - x, y_};
  BaseField y = BaseField::SumOfProductsSerial(lefts, rights);

  // ZZ3 = V
  BaseField zz = std::move(v);

  // ZZZ3 = W
  BaseField zzz = std::move(w);

  return {std::move(x), std::move(y), std::move(zz), std::move(zzz)};
}

#undef CLASS

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_IMPL_H_
