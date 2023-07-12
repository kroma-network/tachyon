#ifndef TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_CURVE_CONFIG_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_CURVE_CONFIG_H_

#include "tachyon/export.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fq.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/sw_curve_config.h"

namespace tachyon {
namespace math {
namespace bn254 {

class TACHYON_EXPORT CurveConfig : public SWCurveConfig<Fq, Fr> {
 public:
  using Config = SWCurveConfig<Fq, Fr>;

  static void Init();
};

using G1AffinePoint = AffinePoint<CurveConfig::Config>;
using G1JacobianPoint = JacobianPoint<CurveConfig::Config>;

}  // namespace bn254
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_CURVE_CONFIG_H_
