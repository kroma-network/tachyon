#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_SW_CURVE_CONFIG_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_SW_CURVE_CONFIG_H_

#include <type_traits>

#include "tachyon/base/no_destructor.h"
#include "tachyon/base/static_storage.h"
#include "tachyon/math/elliptic_curves/jacobian_point.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"

namespace tachyon {
namespace math {

// Config for Short Weierstrass model.
// See https://www.hyperelliptic.org/EFD/g1p/auto-shortw.html for more details.
// This config represents `y² = x³ + a * x + b`, where `a` and `b` are
// constants.
template <typename _BaseField, typename _ScalarField>
class SWCurveConfig {
 public:
  using BaseField = _BaseField;
  using ScalarField = _ScalarField;
  using JacobianPointTy = JacobianPoint<SWCurveConfig<BaseField, ScalarField>>;

  DEFINE_STATIC_STORAGE_TEMPLATE_METHOD(BaseField, A)
  DEFINE_STATIC_STORAGE_TEMPLATE_METHOD(BaseField, B)
  DEFINE_STATIC_STORAGE_TEMPLATE_METHOD(JacobianPointTy, Generator)

  static bool IsOnCurve(const BaseField& x, const BaseField& y) {
    BaseField right = x.Square() * x + B();
    if (!A().IsZero()) {
      right += A() * x;
    }
    return y.Square() == right;
  }

  template <
      typename BaseInputIterator, typename ScalarInputIterator,
      std::enable_if_t<IsAbleToMSM<BaseInputIterator, ScalarInputIterator,
                                   JacobianPointTy, ScalarField>>* = nullptr>
  static JacobianPointTy MSM(BaseInputIterator bases_first,
                             BaseInputIterator bases_last,
                             ScalarInputIterator scalars_first,
                             ScalarInputIterator scalars_last) {
    return VariableBaseMSM<JacobianPointTy>::MSM(
        std::move(bases_first), std::move(bases_last), std::move(scalars_first),
        std::move(scalars_last));
  }

  template <typename BaseContainer, typename ScalarContainer>
  static JacobianPointTy MSM(BaseContainer&& bases, ScalarContainer&& scalars) {
    return MSM(std::begin(std::forward<BaseContainer>(bases)),
               std::end(std::forward<BaseContainer>(bases)),
               std::begin(std::forward<ScalarContainer>(scalars)),
               std::end(std::forward<ScalarContainer>(scalars)));
  }
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_SW_CURVE_CONFIG_H_
