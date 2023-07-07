#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_MSM_UTIL_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_MSM_UTIL_H_

#include <cmath>
#include <type_traits>

#include "tachyon/base/template_util.h"

namespace tachyon {
namespace math {

template <typename BaseInputIterator, typename ScalarInputIterator,
          typename JacobianPoint, typename ScalarField>
inline constexpr bool IsAbleToMSM =
    std::is_same_v<JacobianPoint, base::iter_value_t<BaseInputIterator>>&&
        std::is_same_v<ScalarField, base::iter_value_t<ScalarInputIterator>>;

/// The result of this function is only approximately `ln(a)`
/// [`Explanation of usage`]
///
/// [`Explanation of usage`]:
/// https://github.com/scipr-lab/zexe/issues/79#issue-556220473
constexpr size_t LnWithoutFloats(size_t a) {
  // log2(a) * ln(2)
  return log2(a) * 69 / 100;
}

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_MSM_UTIL_H_
