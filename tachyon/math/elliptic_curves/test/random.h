#ifndef TACHYON_MATH_ELLIPTIC_CURVES_TEST_RANDOM_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_TEST_RANDOM_H_

#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/elliptic_curves/point_conversions.h"

namespace tachyon::math {

template <typename PointTy>
std::vector<PointTy> CreatePseudoRandomPoints(size_t size) {
  // NOTE(chokobole): PointTy::Random() is an expensive operation, which
  // internally, randomly picks a scalar field and multiplies by a
  // generator. In most case, including MSM, bases doesn't affect to a
  // performance or test. So here it just produces pseudo random points by
  // doubling from a first selected random point.
  PointTy p = PointTy::Random();
  return base::CreateVector(size, [&p]() {
    PointTy ret = p;
    p = ConvertPoint<PointTy>(p.Double());
    return ret;
  });
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_TEST_RANDOM_H_
