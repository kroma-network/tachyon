#include "tachyon/math/elliptic_curves/short_weierstrass/test_config.h"

#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"

namespace tachyon {
namespace math {
namespace test {

// static
void CurveConfig::Init() {
  B() = GF7(5);
  Generator() = JacobianPoint<SWCurveConfig<GF7, GF7>>(GF7(5), GF7(5), GF7(1));
}

}  // namespace test
}  // namespace math
}  // namespace tachyon
