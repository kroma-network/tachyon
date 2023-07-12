#ifndef TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_CURVE_CONFIG_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_CURVE_CONFIG_H_

#include "tachyon/export.h"
#include "tachyon/math/elliptic_curves/bls/bls12_381/fq.h"
#include "tachyon/math/elliptic_curves/bls/bls12_381/fr.h"
#include "tachyon/math/elliptic_curves/msm/glv.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/sw_curve_config.h"

namespace tachyon {
namespace math {
namespace bls12_381 {

class TACHYON_EXPORT CurveConfig : public SWCurveConfig<Fq, Fr> {
 public:
  using Config = SWCurveConfig<Fq, Fr>;

  static void Init();

  static Fq& EndomorphismCoefficient();
  static Fr& Lambda();
  static GLV<CurveConfig>::Coefficients ScalarDecompositionCoefficients();

  static JacobianPoint<Config> Endomorphism(const JacobianPoint<Config>& point);
  static AffinePoint<Config> EndomorphismAffine(
      const AffinePoint<Config>& point);
};

using G1AffinePoint = AffinePoint<CurveConfig::Config>;
using G1JacobianPoint = JacobianPoint<CurveConfig::Config>;

}  // namespace bls12_381
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_CURVE_CONFIG_H_
