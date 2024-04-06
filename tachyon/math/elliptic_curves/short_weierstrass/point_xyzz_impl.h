#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_POINT_XYZZ_IMPL_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_POINT_XYZZ_IMPL_H_

#include <utility>

#include "tachyon/math/elliptic_curves/short_weierstrass/point_xyzz.h"

namespace tachyon::math {

#define CLASS      \
  PointXYZZ<Curve, \
            std::enable_if_t<Curve::kType == CurveType::kShortWeierstrass>>

template <typename Curve>
constexpr CLASS CLASS::Add(const PointXYZZ& other) const {
  if (IsZero()) {
    return other;
  }

  if (other.IsZero()) {
    return *this;
  }

  PointXYZZ ret;
  DoAdd(*this, other, ret);
  return ret;
}

template <typename Curve>
constexpr CLASS& CLASS::AddInPlace(const PointXYZZ& other) {
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
constexpr void CLASS::DoAdd(const PointXYZZ& a, const PointXYZZ& b,
                            PointXYZZ& c) {
  // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-add-2008-s
  // U1 = X1 * ZZ2
  BaseField u1 = a.x_ * b.zz_;

  // U2 = X2 * ZZ1
  BaseField u2 = b.x_ * a.zz_;

  // S1 = Y1 * ZZZ2
  BaseField s1 = a.y_ * b.zzz_;

  // S2 = Y2 * ZZZ1
  BaseField s2 = b.y_ * a.zzz_;

  // P = U2 - U1
  BaseField p = u2 - u1;

  // R = S2 - S1
  BaseField r = s2 - s1;

  if (p.IsZero() && r.IsZero()) {
    if (&a == &c) {
      c.DoubleInPlace();
    } else {
      c = a.Double();
    }
    return;
  }

  // PP = P²
  BaseField pp = p.Square();

  // PPP = P * PP
  BaseField ppp = p * pp;

  // Q = U1 * PP
  BaseField q = u1 * pp;

  // X3 = R² - PPP - 2 * Q
  c.x_ = r.Square();
  c.x_ -= ppp;
  c.x_ -= q.Double();

  // Y3 = R * (Q - X3) - S1 * PPP
  BaseField lefts[] = {std::move(r), -s1};
  BaseField rights[] = {q - c.x_, ppp};
  c.y_ = BaseField::SumOfProductsSerial(lefts, rights);

  // ZZ3 = ZZ1 * ZZ2 * PP
  c.zz_ = a.zz_ * b.zz_;
  c.zz_ *= pp;

  // ZZZ3 = ZZZ1 * ZZZ2 * PPP
  c.zzz_ = a.zzz_ * b.zzz_;
  c.zzz_ *= ppp;
}

template <typename Curve>
constexpr CLASS CLASS::Add(const AffinePoint<Curve>& other) const {
  if (IsZero()) {
    return other.ToXYZZ();
  }

  if (other.IsZero()) {
    return *this;
  }

  PointXYZZ ret;
  DoAdd(*this, other, ret);
  return ret;
}

template <typename Curve>
constexpr CLASS& CLASS::AddInPlace(const AffinePoint<Curve>& other) {
  if (IsZero()) {
    return *this = other.ToXYZZ();
  }

  if (other.IsZero()) {
    return *this;
  }

  DoAdd(*this, other, *this);
  return *this;
}

// static
template <typename Curve>
constexpr void CLASS::DoAdd(const PointXYZZ& a, const AffinePoint<Curve>& b,
                            PointXYZZ& c) {
  // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-madd-2008-s
  // U2 = X2 * ZZ1
  BaseField u2 = b.x() * a.zz_;

  // S2 = Y2 * ZZZ1
  BaseField s2 = b.y() * a.zzz_;

  // P = U2 - X1
  BaseField p = u2 - a.x_;

  // R = S2 - Y1
  BaseField r = s2 - a.y_;

  if (p.IsZero() && r.IsZero()) {
    if (&a == &c) {
      c.DoubleInPlace();
    } else {
      c = a.Double();
    }
    return;
  }

  // PP = P²
  BaseField pp = p.Square();

  // PPP = P * PP
  BaseField ppp = p * pp;

  // Q = X1 * PP
  BaseField q = a.x_ * pp;

  // X3 = R² - PPP - 2 * Q
  c.x_ = r.Square();
  c.x_ -= ppp;
  c.x_ -= q.Double();

  // Y3 = R * (Q - X3) - Y1 * PPP
  BaseField lefts[] = {std::move(r), -a.y_};
  BaseField rights[] = {q - c.x_, ppp};
  c.y_ = BaseField::SumOfProductsSerial(lefts, rights);

  // ZZ3 = ZZ1 * PP
  c.zz_ = a.zz_ * pp;

  // ZZZ3 = ZZZ1 * PPP
  c.zzz_ = a.zzz_ * ppp;
}

template <typename Curve>
constexpr CLASS& CLASS::DoubleInPlace() {
  if (IsZero()) {
    return *this;
  }

  // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#doubling-dbl-2008-s-1
  // U = 2 * Y1
  BaseField u = y_.Double();

  // V = U²
  BaseField v = u.Square();

  // W = U * V
  BaseField w = u * v;

  // S = X1 * V
  BaseField s = x_ * v;

  // M = 3 * X1² + a * ZZ1²
  BaseField m = x_.Square();
  m += m.Double();
  if constexpr (!Curve::Config::kAIsZero) {
    m += Curve::Config::MulByA(zz_.Square());
  }

  // X3 = M² - 2 * S
  x_ = m.Square();
  x_ -= s.Double();

  // Y3 = M * (S - X3) - W * Y1
  BaseField lefts[] = {std::move(m), -w};
  BaseField rights[] = {s - x_, y_};
  y_ = BaseField::SumOfProductsSerial(lefts, rights);

  // ZZ3 = V * ZZ1
  zz_ *= v;

  // ZZZ3 = W * ZZZ1
  zzz_ *= w;

  return *this;
}

#undef CLASS

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_POINT_XYZZ_IMPL_H_
