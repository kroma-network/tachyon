#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_PIPPENGER_BASE_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_PIPPENGER_BASE_H_

#include "tachyon/math/base/semigroups.h"

namespace tachyon::math {

template <typename PointTy,
          typename Bucket_ =
              typename internal::AdditiveSemigroupTraits<PointTy>::ReturnTy>
class PippengerBase {
 public:
  using Bucket = Bucket_;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_PIPPENGER_BASE_H_
