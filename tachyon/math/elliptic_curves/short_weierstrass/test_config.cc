#include "tachyon/math/elliptic_curves/short_weierstrass/test_config.h"

#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"

namespace tachyon {
namespace math {
namespace test {

// static
void CurveConfig::Init() {
  GF7::Config::Init();

  B() = GF7Gmp(5);
  Generator() = JacobianPoint<SWCurveConfig<GF7Gmp, GF7Gmp>>(
      GF7Gmp(5), GF7Gmp(5), GF7Gmp(1));
}

}  // namespace test
}  // namespace math
}  // namespace tachyon
