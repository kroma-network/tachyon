#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_TEST_SW_CURVE_CONFIG_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_TEST_SW_CURVE_CONFIG_H_

#include "tachyon/math/elliptic_curves/short_weierstrass/sw_curve_config.h"
#include "tachyon/math/finite_fields/prime_field.h"

namespace tachyon {
namespace math {

class TestSwCurveConfig : public SWCurveConfig<Fp7, Fp7> {
 public:
  using Config = SWCurveConfig<Fp7, Fp7>;

  static void Init();
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_TEST_SW_CURVE_CONFIG_H_
