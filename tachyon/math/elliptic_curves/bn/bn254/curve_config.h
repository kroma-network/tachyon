#ifndef TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_CURVE_CONFIG_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_CURVE_CONFIG_H_

#include "tachyon/math/elliptic_curves/bn/bn254/fq.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/sw_curve_config.h"

namespace tachyon {
namespace math {
namespace bn254 {

template <typename Fq, typename Fr>
class CurveConfig : public SWCurveConfig<Fq, Fr> {
 public:
  using Config = SWCurveConfig<Fq, Fr>;

  static void Init() {
    Fq::Config::Init();
    Config::B() = Fq(3);

    Config::Generator() =
        JacobianPoint<Config>::FromAffine(AffinePoint<Config>(Fq(1), Fq(2)));
  }
};

using G1AffinePoint = AffinePoint<CurveConfig::Config<Fq, Fr>>;
using G1JacobianPoint = JacobianPoint<CurveConfig::Config<Fq, Fr>>;
#if defined(TACHYON_GMP_BACKEND)
using G1AffinePointGmp = AffinePoint<CurveConfig::Config<FqGmp, FrGmp>>;
using G1JacobianPointGmp = JacobianPoint<CurveConfig::Config<FqGmp, FrGmp>>;
#endif  // defined(TACHYON_GMP_BACKEND)

}  // namespace bn254
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_CURVE_CONFIG_H_
