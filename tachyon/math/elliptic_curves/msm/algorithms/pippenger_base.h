#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_PIPPENGER_BASE_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_PIPPENGER_BASE_H_

#include "tachyon/math/base/semigroups.h"
#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/math/elliptic_curves/point_xyzz.h"

namespace tachyon::math {

template <typename PointTy>
class PippengerTraits {
 public:
  using Bucket = typename internal::AdditiveSemigroupTraits<PointTy>::ReturnTy;
};

template <typename Curve>
class PippengerTraits<AffinePoint<Curve>> {
 public:
  using Bucket = PointXYZZ<Curve>;
};

template <typename PointTy,
          typename Bucket_ = typename PippengerTraits<PointTy>::Bucket>
class PippengerBase {
 public:
  using Bucket = Bucket_;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_PIPPENGER_BASE_H_
