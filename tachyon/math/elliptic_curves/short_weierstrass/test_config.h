#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_TEST_SW_CURVE_CONFIG_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_TEST_SW_CURVE_CONFIG_H_

#include "tachyon/math/elliptic_curves/short_weierstrass/sw_curve_config.h"
#include "tachyon/math/finite_fields/prime_field.h"

namespace tachyon {
namespace math {
namespace test {

class CurveConfig : public SWCurveConfig<GF7Gmp, GF7Gmp> {
 public:
  using Config = SWCurveConfig<GF7Gmp, GF7Gmp>;

  static void Init();
};

}  // namespace test
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_TEST_SW_CURVE_CONFIG_H_
