#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_SW_CURVE_CUDA_CU_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_SW_CURVE_CUDA_CU_H_

#include "tachyon/math/elliptic_curves/short_weierstrass/sw_curve_base.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/sw_curve_config_traits.h"
#include "tachyon/math/geometry/point2.h"

namespace tachyon::math {

template <typename _Config>
class SWCurveCuda : public SWCurveBase<SWCurveCuda<_Config>> {
 public:
  using Config = _Config;
  using BaseField = typename Config::BaseField;
  using ScalarField = typename Config::ScalarField;
  using AffinePointTy = AffinePoint<SWCurveCuda<Config>>;
  using ProjectivePointTy = ProjectivePoint<SWCurveCuda<Config>>;
  using JacobianPointTy = JacobianPoint<SWCurveCuda<Config>>;
  using PointXYZZTy = PointXYZZ<SWCurveCuda<Config>>;

  constexpr static bool kIsSWCurve = true;

  constexpr static BaseField A() { return BaseField::FromMontgomery(GetA()); }

  constexpr static BaseField B() { return BaseField::FromMontgomery(GetB()); }

  constexpr static JacobianPointTy Generator() {
    return JacobianPointTy(BaseField::FromMontgomery(GetGenerator().x),
                           BaseField::FromMontgomery(GetGenerator().y),
                           BaseField::One());
  }

  static void Init() {
    // Do nothing.
  }

 private:
  constexpr static auto GetA() { return Config::kA; }

  constexpr static auto GetB() { return Config::kB; }

  constexpr static auto GetGenerator() { return Config::kGenerator; }
};

template <typename Config>
struct SWCurveConfigTraits<SWCurveCuda<Config>> {
  using BaseField = typename Config::BaseField;
  using ScalarField = typename Config::ScalarField;
  using AffinePointTy = AffinePoint<SWCurveCuda<Config>>;
  using ProjectivePointTy = ProjectivePoint<SWCurveCuda<Config>>;
  using JacobianPointTy = JacobianPoint<SWCurveCuda<Config>>;
  using PointXYZZTy = PointXYZZ<SWCurveCuda<Config>>;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_SW_CURVE_CUDA_CU_H_
