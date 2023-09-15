#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_SW_CURVE_BASE_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_SW_CURVE_BASE_H_

#include "tachyon/math/elliptic_curves/short_weierstrass/sw_curve_traits.h"

namespace tachyon::math {

// Config for Short Weierstrass model.
// See https://www.hyperelliptic.org/EFD/g1p/auto-shortw.html for more details.
// This config represents `y² = x³ + a * x + b`, where `a` and `b` are
// constants.
template <typename SWCurveConfig>
class SWCurveBase {
 public:
  using BaseField = typename SWCurveTraits<SWCurveConfig>::BaseField;
  using ScalarField = typename SWCurveTraits<SWCurveConfig>::ScalarField;
  using AffinePointTy = typename SWCurveTraits<SWCurveConfig>::AffinePointTy;
  using ProjectivePointTy =
      typename SWCurveTraits<SWCurveConfig>::ProjectivePointTy;
  using JacobianPointTy =
      typename SWCurveTraits<SWCurveConfig>::JacobianPointTy;
  using PointXYZZTy = typename SWCurveTraits<SWCurveConfig>::PointXYZZTy;

  constexpr static bool IsOnCurve(const AffinePointTy& point) {
    if (point.infinity()) return false;
    BaseField right = point.x().Square() * point.x() + SWCurveConfig::B();
    if (!SWCurveConfig::A().IsZero()) {
      right += SWCurveConfig::A() * point.x();
    }
    return point.y().Square() == right;
  }

  constexpr static bool IsOnCurve(const ProjectivePointTy& point) {
    if (point.z().IsZero()) return false;
    BaseField z2 = point.z().Square();
    BaseField z3 = z2 * point.z();
    BaseField right = point.x().Square() * point.x() + SWCurveConfig::B() * z3;
    if (!SWCurveConfig::A().IsZero()) {
      right += SWCurveConfig::A() * point.x() * z2;
    }
    return point.y().Square() * point.z() == right;
  }

  constexpr static bool IsOnCurve(const JacobianPointTy& point) {
    if (point.z().IsZero()) return false;
    BaseField z3 = point.z().Square() * point.z();
    BaseField right =
        point.x().Square() * point.x() + SWCurveConfig::B() * z3.Square();
    if (!SWCurveConfig::A().IsZero()) {
      right += SWCurveConfig::A() * point.x() * z3 * point.z();
    }
    return point.y().Square() == right;
  }

  constexpr static bool IsOnCurve(const PointXYZZTy& point) {
    if (point.zzz().IsZero()) return false;
    BaseField right = point.x().Square() * point.x() +
                      SWCurveConfig::B() * point.zzz().Square();
    if (!SWCurveConfig::A().IsZero()) {
      right += SWCurveConfig::A() * point.x() * point.zz().Square();
    }
    return point.y().Square() == right;
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_SW_CURVE_BASE_H_
