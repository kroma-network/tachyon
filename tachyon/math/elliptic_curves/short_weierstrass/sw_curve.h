#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_SW_CURVE_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_SW_CURVE_H_

#include "tachyon/base/static_storage.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/sw_curve_base.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/sw_curve_config_traits.h"

namespace tachyon {
namespace math {

template <typename _Config>
class SWCurve : public SWCurveBase<SWCurve<_Config>> {
 public:
  using Config = _Config;
  using BaseField = typename Config::BaseField;
  using ScalarField = typename Config::ScalarField;
  using AffinePointTy = AffinePoint<SWCurve<Config>>;
  using ProjectivePointTy = ProjectivePoint<SWCurve<Config>>;
  using JacobianPointTy = JacobianPoint<SWCurve<Config>>;

  constexpr static bool kIsSWCurve = true;

  DEFINE_STATIC_STORAGE_TEMPLATE_METHOD(BaseField, A)
  DEFINE_STATIC_STORAGE_TEMPLATE_METHOD(BaseField, B)
  DEFINE_STATIC_STORAGE_TEMPLATE_METHOD(JacobianPointTy, Generator)

  static void Init() {
    BaseField::Init();
    ScalarField::Init();

    A() = BaseField::FromMontgomery(Config::kA);
    B() = BaseField::FromMontgomery(Config::kB);
    Generator() = JacobianPointTy(
        BaseField::FromMontgomery(Config::kGenerator.x),
        BaseField::FromMontgomery(Config::kGenerator.y), BaseField::One());
  }
};

template <typename Config>
struct SWCurveConfigTraits<SWCurve<Config>> {
  using BaseField = typename Config::BaseField;
  using ScalarField = typename Config::ScalarField;
  using AffinePointTy = AffinePoint<SWCurve<Config>>;
  using ProjectivePointTy = ProjectivePoint<SWCurve<Config>>;
  using JacobianPointTy = JacobianPoint<SWCurve<Config>>;
  using PointXYZZTy = PointXYZZ<SWCurve<Config>>;
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_SW_CURVE_H_
