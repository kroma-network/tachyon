#include "tachyon/math/elliptic_curves/short_weierstrass/test_config.h"

#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"

namespace tachyon {
namespace math {

// static
void TestSwCurveConfig::Init() {
  B() = Fp7(5);
  Generator() = JacobianPoint<SWCurveConfig<Fp7, Fp7>>(Fp7(5), Fp7(5), Fp7(1));
}

}  // namespace math
}  // namespace tachyon
