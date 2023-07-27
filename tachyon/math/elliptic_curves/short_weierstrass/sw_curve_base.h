#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_SW_CURVE_BASE_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_SW_CURVE_BASE_H_

#include <type_traits>

#include "tachyon/math/elliptic_curves/jacobian_point.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"
#include "tachyon/math/elliptic_curves/point_xyzz.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/sw_curve_config_traits.h"

namespace tachyon {
namespace math {

// Config for Short Weierstrass model.
// See https://www.hyperelliptic.org/EFD/g1p/auto-shortw.html for more details.
// This config represents `y² = x³ + a * x + b`, where `a` and `b` are
// constants.
template <typename SWCurveConfig>
class SWCurveBase {
 public:
  using BaseField = typename SWCurveConfigTraits<SWCurveConfig>::BaseField;
  using ScalarField = typename SWCurveConfigTraits<SWCurveConfig>::ScalarField;
  using AffinePointTy =
      typename SWCurveConfigTraits<SWCurveConfig>::AffinePointTy;
  using JacobianPointTy =
      typename SWCurveConfigTraits<SWCurveConfig>::JacobianPointTy;
  using PointXYZZTy = typename SWCurveConfigTraits<SWCurveConfig>::PointXYZZTy;

  constexpr static bool IsOnCurve(const AffinePointTy& point) {
    if (point.infinity()) return false;
    BaseField right = point.x().Square() * point.x() + SWCurveConfig::B();
    if (!SWCurveConfig::A().IsZero()) {
      right += SWCurveConfig::A() * point.x();
    }
    return point.y().Square() == right;
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

  template <
      typename BaseInputIterator, typename ScalarInputIterator,
      std::enable_if_t<IsAbleToMSM<BaseInputIterator, ScalarInputIterator,
                                   JacobianPointTy, ScalarField>>* = nullptr>
  static JacobianPointTy MSM(BaseInputIterator bases_first,
                             BaseInputIterator bases_last,
                             ScalarInputIterator scalars_first,
                             ScalarInputIterator scalars_last) {
    return VariableBaseMSM<JacobianPointTy>::MSM(
        std::move(bases_first), std::move(bases_last), std::move(scalars_first),
        std::move(scalars_last));
  }

  template <typename BaseContainer, typename ScalarContainer>
  static JacobianPointTy MSM(BaseContainer&& bases, ScalarContainer&& scalars) {
    return MSM(std::begin(std::forward<BaseContainer>(bases)),
               std::end(std::forward<BaseContainer>(bases)),
               std::begin(std::forward<ScalarContainer>(scalars)),
               std::end(std::forward<ScalarContainer>(scalars)));
  }
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_SW_CURVE_BASE_H_
