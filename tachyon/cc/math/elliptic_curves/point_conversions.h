#ifndef TACHYON_CC_MATH_ELLIPTIC_CURVES_POINT_CONVERSIONS_H_
#define TACHYON_CC_MATH_ELLIPTIC_CURVES_POINT_CONVERSIONS_H_

#include <stddef.h>

#include <type_traits>

#include "tachyon/cc/math/elliptic_curves/point_traits_forward.h"
#include "tachyon/cc/math/finite_fields/prime_field_conversions.h"
#include "tachyon/math/geometry/point2.h"
#include "tachyon/math/geometry/point3.h"
#include "tachyon/math/geometry/point4.h"

namespace tachyon::cc::math {

template <typename CPoint,
          typename CurvePoint = typename PointTraits<CPoint>::CurvePoint>
const CurvePoint& native_cast(const CPoint& point_in) {
  static_assert(sizeof(CurvePoint) == sizeof(CPoint));
  return reinterpret_cast<const CurvePoint&>(point_in);
}

template <typename CPoint,
          typename CurvePoint = typename PointTraits<CPoint>::CurvePoint>
CurvePoint& native_cast(CPoint& point_in) {
  static_assert(sizeof(CurvePoint) == sizeof(CPoint));
  return reinterpret_cast<CurvePoint&>(point_in);
}

template <typename Point,
          typename CPoint = typename PointTraits<Point>::CCurvePoint>
const CPoint& c_cast(const Point& point_in) {
  static_assert(sizeof(CPoint) == sizeof(Point));
  return reinterpret_cast<const CPoint&>(point_in);
}

template <typename Point,
          typename CPoint = typename PointTraits<Point>::CCurvePoint>
CPoint& c_cast(Point& point_in) {
  static_assert(sizeof(CPoint) == sizeof(Point));
  return reinterpret_cast<CPoint&>(point_in);
}

}  // namespace tachyon::cc::math

#endif  // TACHYON_CC_MATH_ELLIPTIC_CURVES_POINT_CONVERSIONS_H_
