#ifndef TACHYON_C_MATH_ELLIPTIC_CURVES_POINT_CONVERSIONS_H_
#define TACHYON_C_MATH_ELLIPTIC_CURVES_POINT_CONVERSIONS_H_

#include <stddef.h>

#include <type_traits>

#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/math/elliptic_curves/point_traits_forward.h"
#include "tachyon/math/geometry/point2.h"
#include "tachyon/math/geometry/point3.h"
#include "tachyon/math/geometry/point4.h"

namespace tachyon::c::math {

template <typename CPoint,
          typename CurvePoint = typename PointTraits<CPoint>::CurvePoint,
          typename BaseField = typename CurvePoint::BaseField>
CurvePoint ToAffinePoint(const CPoint& point_in) {
  tachyon::math::Point2<BaseField> point;
  point.x = c::base::native_cast(point_in.x);
  point.y = c::base::native_cast(point_in.y);
  return CurvePoint(point);
}

template <typename CPoint,
          typename CurvePoint = typename PointTraits<CPoint>::CurvePoint,
          typename BaseField = typename CurvePoint::BaseField>
CurvePoint ToProjectivePoint(const CPoint& point_in) {
  tachyon::math::Point3<BaseField> point;
  point.x = c::base::native_cast(point_in.x);
  point.y = c::base::native_cast(point_in.y);
  point.z = c::base::native_cast(point_in.z);
  return CurvePoint(point);
}

template <typename CPoint,
          typename CurvePoint = typename PointTraits<CPoint>::CurvePoint,
          typename BaseField = typename CurvePoint::BaseField>
CurvePoint ToJacobianPoint(const CPoint& point_in) {
  tachyon::math::Point3<BaseField> point;
  point.x = c::base::native_cast(point_in.x);
  point.y = c::base::native_cast(point_in.y);
  point.z = c::base::native_cast(point_in.z);
  return CurvePoint(point);
}

template <typename CPoint,
          typename CurvePoint = typename PointTraits<CPoint>::CurvePoint,
          typename BaseField = typename CurvePoint::BaseField>
CurvePoint ToPointXYZZ(const CPoint& point_in) {
  tachyon::math::Point4<BaseField> point;
  point.x = c::base::native_cast(point_in.x);
  point.y = c::base::native_cast(point_in.y);
  point.z = c::base::native_cast(point_in.zz);
  point.w = c::base::native_cast(point_in.zzz);
  return CurvePoint(point);
}

template <typename CPoint, typename Point = typename PointTraits<CPoint>::Point,
          typename BaseField = typename Point::value_type>
Point ToPoint2(const CPoint& point_in) {
  Point point;
  point.x = c::base::native_cast(point_in.x);
  point.y = c::base::native_cast(point_in.y);
  return point;
}

template <typename CPoint, typename Point = typename PointTraits<CPoint>::Point,
          typename BaseField = typename Point::value_type>
Point ToPoint3(const CPoint& point_in) {
  Point point;
  point.x = c::base::native_cast(point_in.x);
  point.y = c::base::native_cast(point_in.y);
  point.z = c::base::native_cast(point_in.z);
  return point;
}

template <typename CPoint, typename Point = typename PointTraits<CPoint>::Point,
          typename BaseField = typename Point::value_type>
Point ToPoint4(const CPoint& point_in) {
  Point point;
  point.x = c::base::native_cast(point_in.x);
  point.y = c::base::native_cast(point_in.y);
  point.z = c::base::native_cast(point_in.z);
  point.w = c::base::native_cast(point_in.w);
  return point;
}

template <typename Point,
          typename CPoint = typename PointTraits<Point>::CCurvePoint>
CPoint ToCAffinePoint(const Point& point_in) {
  CPoint ret;
  ret.x = c::base::c_cast(point_in.x());
  ret.y = c::base::c_cast(point_in.y());
  return ret;
}

template <typename Point,
          typename CPoint = typename PointTraits<Point>::CCurvePoint>
CPoint ToCProjectivePoint(const Point& point_in) {
  CPoint ret;
  ret.x = c::base::c_cast(point_in.x());
  ret.y = c::base::c_cast(point_in.y());
  ret.z = c::base::c_cast(point_in.z());
  return ret;
}

template <typename Point,
          typename CPoint = typename PointTraits<Point>::CCurvePoint>
CPoint ToCJacobianPoint(const Point& point_in) {
  CPoint ret;
  ret.x = c::base::c_cast(point_in.x());
  ret.y = c::base::c_cast(point_in.y());
  ret.z = c::base::c_cast(point_in.z());
  return ret;
}

template <typename Point,
          typename CPoint = typename PointTraits<Point>::CCurvePoint>
CPoint ToCPointXYZZ(const Point& point_in) {
  CPoint ret;
  ret.x = c::base::c_cast(point_in.x());
  ret.y = c::base::c_cast(point_in.y());
  ret.zz = c::base::c_cast(point_in.zz());
  ret.zzz = c::base::c_cast(point_in.zzz());
  return ret;
}

template <typename Point, typename CPoint>
void ToCPoint2(const Point& point_in, CPoint* point_out) {
  point_out->x = c::base::c_cast(point_in.x());
  point_out->y = c::base::c_cast(point_in.y());
}

template <typename Point, typename CPoint>
void ToCPoint3(const Point& point_in, CPoint* point_out) {
  point_out->x = c::base::c_cast(point_in.x());
  point_out->y = c::base::c_cast(point_in.y());
  point_out->z = c::base::c_cast(point_in.z());
}

template <typename Point, typename CPoint>
void ToCPoint4(const Point& point_in, CPoint* point_out) {
  point_out->x = c::base::c_cast(point_in.x());
  point_out->y = c::base::c_cast(point_in.y());
  point_out->z = c::base::c_cast(point_in.z());
  point_out->w = c::base::c_cast(point_in.w());
}

}  // namespace tachyon::c::math

#endif  // TACHYON_C_MATH_ELLIPTIC_CURVES_POINT_CONVERSIONS_H_
