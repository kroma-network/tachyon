#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_PIPPENGER_GPU_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_PIPPENGER_GPU_H_

#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/math/elliptic_curves/msm/algorithms/pippenger_base.h"
#include "tachyon/math/elliptic_curves/point_xyzz.h"

namespace tachyon::math {

template <typename Curve>
class PippengerGpu
    : public PippengerBase<AffinePoint<Curve>, PointXYZZ<Curve>> {};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_PIPPENGER_GPU_H_
