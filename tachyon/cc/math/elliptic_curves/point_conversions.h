#ifndef TACHYON_CC_MATH_ELLIPTIC_CURVES_POINT_CONVERSIONS_H_
#define TACHYON_CC_MATH_ELLIPTIC_CURVES_POINT_CONVERSIONS_H_

#include <stddef.h>

namespace tachyon::math {

template <typename CPointTy, typename PointTy,
          typename BaseField = typename PointTy::BaseField,
          size_t LimbNums = BaseField::kLimbNums>
CPointTy* CreateCPoint3Ptr(const PointTy& point) {
  CPointTy* ret = new CPointTy;
  memcpy(&ret->x, point.x().value().limbs, sizeof(uint64_t) * LimbNums);
  memcpy(&ret->y, point.y().value().limbs, sizeof(uint64_t) * LimbNums);
  memcpy(&ret->z, point.z().value().limbs, sizeof(uint64_t) * LimbNums);
  return ret;
}

}  // namespace tachyon::math

#endif  // TACHYON_CC_MATH_ELLIPTIC_CURVES_POINT_CONVERSIONS_H_
