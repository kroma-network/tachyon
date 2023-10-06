#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_POINT_XYZZ_IMPL_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_POINT_XYZZ_IMPL_H_

#include <utility>

#include "tachyon/math/elliptic_curves/short_weierstrass/point_xyzz.h"

namespace tachyon::math {

#define CLASS PointXYZZ<Curve, std::enable_if_t<Curve::kIsSWCurve>>

template <typename Curve>
constexpr CLASS& CLASS::AddInPlace(const PointXYZZ& other) {
  if (IsZero()) {
    return *this = other;
  }

  if (other.IsZero()) {
    return *this;
  }

  // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-add-2008-s
  // U1 = X1 * ZZ2
  BaseField u1 = x_;
  u1 *= other.zz_;

  // U2 = X2 * ZZ1
  BaseField u2 = other.x_;
  u2 *= zz_;

  // S1 = Y1 * ZZZ2
  BaseField s1 = y_;
  s1 *= other.zzz_;

  // S2 = Y2 * ZZZ1
  BaseField s2 = other.y_;
  s2 *= zzz_;

  // P = U2 - U1
  BaseField p = std::move(u2);
  p -= u1;

  // R = S2 - S1
  BaseField r = std::move(s2);
  r -= s1;

  if (p.IsZero() && r.IsZero()) {
    return DoubleInPlace();
  }

  // PP = P²
  BaseField pp = p;
  pp.SquareInPlace();

  // PPP = P * PP
  BaseField ppp = std::move(p);
  ppp *= pp;

  // Q = U1 * PP
  BaseField q = std::move(u1);
  q *= pp;

  // X3 = R² - PPP - 2 * Q
  x_ = r;
  x_.SquareInPlace();
  x_ -= ppp;
  x_ -= q.Double();

  // Y3 = R * (Q - X3) - S1 * PPP
  BaseField lefts[] = {std::move(r), -s1};
  BaseField rights[] = {q - x_, ppp};
  y_ = BaseField::SumOfProducts(lefts, rights);

  // ZZ3 = ZZ1 * ZZ2 * PP
  zz_ *= other.zz_;
  zz_ *= pp;

  // ZZZ3 = ZZZ1 * ZZZ2 * PPP
  zzz_ *= other.zzz_;
  zzz_ *= ppp;

  return *this;
}

template <typename Curve>
constexpr CLASS& CLASS::AddInPlace(const AffinePoint<Curve>& other) {
  if (IsZero()) {
    return *this = other.ToXYZZ();
  }

  if (other.IsZero()) {
    return *this;
  }

  // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-madd-2008-s
  // U2 = X2 * ZZ1
  BaseField u2 = other.x();
  u2 *= zz_;

  // S2 = Y2 * ZZZ1
  BaseField s2 = other.y();
  s2 *= zzz_;

  // P = U2 - X1
  BaseField p = std::move(u2);
  p -= x_;

  // R = S2 - Y1
  BaseField r = std::move(s2);
  r -= y_;

  if (p.IsZero() && r.IsZero()) {
    return DoubleInPlace();
  }

  // PP = P²
  BaseField pp = p;
  pp.SquareInPlace();

  // PPP = P * PP
  BaseField ppp = std::move(p);
  ppp *= pp;

  // Q = X1 * PP
  BaseField q = std::move(x_);
  q *= pp;

  // X3 = R² - PPP - 2 * Q
  x_ = r;
  x_.SquareInPlace();
  x_ -= ppp;
  x_ -= q.Double();

  // Y3 = R * (Q - X3) - Y1 * PPP
  BaseField lefts[] = {std::move(r), -y_};
  BaseField rights[] = {q - x_, ppp};
  y_ = BaseField::SumOfProducts(lefts, rights);

  // ZZ3 = ZZ1 * PP
  zz_ *= pp;

  // ZZZ3 = ZZZ1 * PPP
  zzz_ *= ppp;

  return *this;
}

template <typename Curve>
constexpr CLASS& CLASS::DoubleInPlace() {
  if (IsZero()) {
    return *this;
  }

  // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#doubling-dbl-2008-s-1
  // U = 2 * Y1
  BaseField u = y_;
  u.DoubleInPlace();

  // V = U²
  BaseField v = u;
  v.SquareInPlace();

  // W = U * V
  BaseField w = std::move(u);
  w *= v;

  // S = X1 * V
  BaseField s = x_;
  s *= v;

  // M = 3 * X1² + a * ZZ1²
  BaseField m = std::move(x_);
  m.SquareInPlace();
  m += m.Double();
  if constexpr (!Curve::Config::kAIsZero) {
    m += Curve::Config::MulByA(zz_.Square());
  }

  // X3 = M² - 2 * S
  x_ = m;
  x_.SquareInPlace();
  x_ -= s.Double();

  // Y3 = M * (S - X3) - W * Y1
  BaseField lefts[] = {std::move(m), -w};
  BaseField rights[] = {s - x_, y_};
  y_ = BaseField::SumOfProducts(lefts, rights);

  // ZZ3 = V * ZZ1
  zz_ *= v;

  // ZZZ3 = W * ZZZ1
  zzz_ *= w;

  return *this;
}

#undef CLASS

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_POINT_XYZZ_IMPL_H_
