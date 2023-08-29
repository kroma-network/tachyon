#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_MSM_UTIL_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_MSM_UTIL_H_

#include <cmath>
#include <type_traits>

#include "tachyon/base/template_util.h"

namespace tachyon::math {

template <typename BaseInputIterator, typename ScalarInputIterator,
          typename PointTy, typename ScalarField>
inline constexpr bool IsAbleToMSM =
    std::is_same_v<PointTy, base::iter_value_t<BaseInputIterator>> &&
    std::is_same_v<ScalarField, base::iter_value_t<ScalarInputIterator>>;

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_MSM_UTIL_H_
