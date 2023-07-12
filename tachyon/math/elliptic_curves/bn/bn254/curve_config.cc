#include "tachyon/math/elliptic_curves/bn/bn254/curve_config.h"

#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"

namespace tachyon {
namespace math {
namespace bn254 {

// static
void CurveConfig::Init() {
  B() = Fq(3);

  Generator() =
      JacobianPoint<Config>::FromAffine(AffinePoint<Config>(Fq(1), Fq(2)));
}

}  // namespace bn254
}  // namespace math
}  // namespace tachyon
