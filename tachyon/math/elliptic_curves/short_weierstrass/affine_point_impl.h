#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_IMPL_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_IMPL_H_

#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"

namespace tachyon {
namespace math {

#define CLASS AffinePoint<Curve, std::enable_if_t<Curve::kIsSWCurve>>

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
  BaseField y = std::move(s);
  y -= x;
  std::array<BaseField, 2> a;
  a[0] = std::move(m);
  a[1] = -w;
  std::array<BaseField, 2> b;
  b[0] = y;
  b[1] = y_;
  y = BaseField::SumOfProducts(a, b);

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
