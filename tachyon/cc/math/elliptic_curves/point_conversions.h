#ifndef TACHYON_CC_MATH_ELLIPTIC_CURVES_POINT_CONVERSIONS_H_
#define TACHYON_CC_MATH_ELLIPTIC_CURVES_POINT_CONVERSIONS_H_

#include <stddef.h>

#include <type_traits>

#include "tachyon/cc/math/elliptic_curves/point_traits.h"
#include "tachyon/cc/math/finite_fields/prime_field_conversions.h"
#include "tachyon/math/geometry/point2.h"
#include "tachyon/math/geometry/point3.h"
#include "tachyon/math/geometry/point4.h"

namespace tachyon::cc::math {

template <typename CPoint,
          typename CurvePoint = typename PointTraits<CPoint>::CurvePoint,
          typename BaseField = typename CurvePoint::BaseField>
CurvePoint ToAffinePoint(const CPoint& point_in) {
  tachyon::math::Point2<BaseField> point;
  point.x = BaseField::FromMontgomery(ToBigInt(point_in.x));
  point.y = BaseField::FromMontgomery(ToBigInt(point_in.y));
  return CurvePoint(point, point_in.infinity);
}

template <typename CPoint,
          typename CurvePoint = typename PointTraits<CPoint>::CurvePoint,
          typename BaseField = typename CurvePoint::BaseField>
CurvePoint ToProjectivePoint(const CPoint& point_in) {
  tachyon::math::Point3<BaseField> point;
  point.x = BaseField::FromMontgomery(ToBigInt(point_in.x));
  point.y = BaseField::FromMontgomery(ToBigInt(point_in.y));
  point.z = BaseField::FromMontgomery(ToBigInt(point_in.z));
  return CurvePoint(point);
}

template <typename CPoint,
          typename CurvePoint = typename PointTraits<CPoint>::CurvePoint,
          typename BaseField = typename CurvePoint::BaseField>
CurvePoint ToJacobianPoint(const CPoint& point_in) {
  tachyon::math::Point3<BaseField> point;
  point.x = BaseField::FromMontgomery(ToBigInt(point_in.x));
  point.y = BaseField::FromMontgomery(ToBigInt(point_in.y));
  point.z = BaseField::FromMontgomery(ToBigInt(point_in.z));
  return CurvePoint(point);
}

template <typename CPoint,
          typename CurvePoint = typename PointTraits<CPoint>::CurvePoint,
          typename BaseField = typename CurvePoint::BaseField>
CurvePoint ToPointXYZZ(const CPoint& point_in) {
  tachyon::math::Point4<BaseField> point;
  point.x = BaseField::FromMontgomery(ToBigInt(point_in.x));
  point.y = BaseField::FromMontgomery(ToBigInt(point_in.y));
  point.z = BaseField::FromMontgomery(ToBigInt(point_in.zz));
  point.w = BaseField::FromMontgomery(ToBigInt(point_in.zzz));
  return CurvePoint(point);
}

template <typename CPoint, typename Point = typename PointTraits<CPoint>::Point,
          typename BaseField = typename Point::value_type>
Point ToPoint2(const CPoint& point_in) {
  Point point;
  point.x = BaseField::FromMontgomery(ToBigInt(point_in.x));
  point.y = BaseField::FromMontgomery(ToBigInt(point_in.y));
  return point;
}

template <typename CPoint, typename Point = typename PointTraits<CPoint>::Point,
          typename BaseField = typename Point::value_type>
Point ToPoint3(const CPoint& point_in) {
  Point point;
  point.x = BaseField::FromMontgomery(ToBigInt(point_in.x));
  point.y = BaseField::FromMontgomery(ToBigInt(point_in.y));
  point.z = BaseField::FromMontgomery(ToBigInt(point_in.z));
  return point;
}

template <typename CPoint, typename Point = typename PointTraits<CPoint>::Point,
          typename BaseField = typename Point::value_type>
Point ToPoint4(const CPoint& point_in) {
  Point point;
  point.x = BaseField::FromMontgomery(ToBigInt(point_in.x));
  point.y = BaseField::FromMontgomery(ToBigInt(point_in.y));
  point.z = BaseField::FromMontgomery(ToBigInt(point_in.z));
  point.w = BaseField::FromMontgomery(ToBigInt(point_in.w));
  return point;
}

template <typename Point,
          typename CPoint = typename PointTraits<Point>::CCurvePoint,
          typename BaseField = typename Point::BaseField,
          size_t LimbNumbs = BaseField::kLimbNums>
CPoint ToCAffinePoint(const Point& point_in) {
  CPoint ret;
  memcpy(ret.x.limbs, point_in.x().value().limbs, sizeof(uint64_t) * LimbNumbs);
  memcpy(ret.y.limbs, point_in.y().value().limbs, sizeof(uint64_t) * LimbNumbs);
  ret.infinity = point_in.infinity();
  return ret;
}

template <typename Point,
          typename CPoint = typename PointTraits<Point>::CCurvePoint,
          typename BaseField = typename Point::BaseField,
          size_t LimbNumbs = BaseField::kLimbNums>
CPoint ToCProjectivePoint(const Point& point_in) {
  CPoint ret;
  memcpy(ret.x.limbs, point_in.x().value().limbs, sizeof(uint64_t) * LimbNumbs);
  memcpy(ret.y.limbs, point_in.y().value().limbs, sizeof(uint64_t) * LimbNumbs);
  memcpy(ret.z.limbs, point_in.z().value().limbs, sizeof(uint64_t) * LimbNumbs);
  return ret;
}

template <typename Point,
          typename CPoint = typename PointTraits<Point>::CCurvePoint,
          typename BaseField = typename Point::BaseField,
          size_t LimbNumbs = BaseField::kLimbNums>
CPoint ToCJacobianPoint(const Point& point_in) {
  CPoint ret;
  memcpy(ret.x.limbs, point_in.x().value().limbs, sizeof(uint64_t) * LimbNumbs);
  memcpy(ret.y.limbs, point_in.y().value().limbs, sizeof(uint64_t) * LimbNumbs);
  memcpy(ret.z.limbs, point_in.z().value().limbs, sizeof(uint64_t) * LimbNumbs);
  return ret;
}

template <typename Point,
          typename CPoint = typename PointTraits<Point>::CCurvePoint,
          typename BaseField = typename Point::BaseField,
          size_t LimbNumbs = BaseField::kLimbNums>
CPoint ToCPointXYZZ(const Point& point_in) {
  CPoint ret;
  memcpy(ret.x.limbs, point_in.x().value().limbs, sizeof(uint64_t) * LimbNumbs);
  memcpy(ret.y.limbs, point_in.y().value().limbs, sizeof(uint64_t) * LimbNumbs);
  memcpy(ret.zz.limbs, point_in.zz().value().limbs,
         sizeof(uint64_t) * LimbNumbs);
  memcpy(ret.zzz.limbs, point_in.zzz().value().limbs,
         sizeof(uint64_t) * LimbNumbs);
  return ret;
}

template <typename Point, typename CPoint,
          typename BaseField = typename Point::BaseField,
          size_t LimbNumbs = BaseField::kLimbNums>
void ToCPoint2(const Point& point_in, CPoint* point_out) {
  memcpy(point_out->x.limbs, point_in.x().value().limbs,
         sizeof(uint64_t) * LimbNumbs);
  memcpy(point_out->y.limbs, point_in.y().value().limbs,
         sizeof(uint64_t) * LimbNumbs);
}

template <typename Point, typename CPoint,
          typename BaseField = typename Point::BaseField,
          size_t LimbNumbs = BaseField::kLimbNums>
void ToCPoint3(const Point& point_in, CPoint* point_out) {
  memcpy(point_out->x.limbs, point_in.x().value().limbs,
         sizeof(uint64_t) * LimbNumbs);
  memcpy(point_out->y.limbs, point_in.y().value().limbs,
         sizeof(uint64_t) * LimbNumbs);
  memcpy(point_out->z.limbs, point_in.z().value().limbs,
         sizeof(uint64_t) * LimbNumbs);
}

template <typename Point, typename CPoint,
          typename BaseField = typename Point::BaseField,
          size_t LimbNumbs = BaseField::kLimbNums>
void ToCPoint4(const Point& point_in, CPoint* point_out) {
  memcpy(point_out->x.limbs, point_in.x().value().limbs,
         sizeof(uint64_t) * LimbNumbs);
  memcpy(point_out->y.limbs, point_in.y().value().limbs,
         sizeof(uint64_t) * LimbNumbs);
  memcpy(point_out->z.limbs, point_in.z().value().limbs,
         sizeof(uint64_t) * LimbNumbs);
  memcpy(point_out->w.limbs, point_in.w().value().limbs,
         sizeof(uint64_t) * LimbNumbs);
}

}  // namespace tachyon::cc::math

#endif  // TACHYON_CC_MATH_ELLIPTIC_CURVES_POINT_CONVERSIONS_H_
