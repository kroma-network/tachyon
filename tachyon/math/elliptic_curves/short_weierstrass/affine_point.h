#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_H_

#include <type_traits>

#include "absl/strings/substitute.h"

#include "tachyon/base/logging.h"
#include "tachyon/math/base/groups.h"
#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/math/elliptic_curves/curve_config.h"
#include "tachyon/math/elliptic_curves/jacobian_point.h"
#include "tachyon/math/elliptic_curves/point_xyzz.h"
#include "tachyon/math/elliptic_curves/projective_point.h"
#include "tachyon/math/elliptic_curves/semigroups.h"
#include "tachyon/math/geometry/point2.h"

namespace tachyon::math {

template <typename _Curve>
class AffinePoint<_Curve, std::enable_if_t<_Curve::kIsSWCurve>> final
    : public AdditiveGroup<AffinePoint<_Curve>> {
 public:
  constexpr static const bool kNegationIsCheap = true;

  using Curve = _Curve;
  using BaseField = typename Curve::BaseField;
  using ScalarField = typename Curve::ScalarField;
  using ProjectivePointTy = ProjectivePoint<Curve>;
  using JacobianPointTy = JacobianPoint<Curve>;
  using PointXYZZTy = PointXYZZ<Curve>;

  constexpr AffinePoint()
      : AffinePoint(BaseField::Zero(), BaseField::Zero(), true) {}
  constexpr AffinePoint(const Point2<BaseField>& point, bool infinity = false)
      : AffinePoint(point.x, point.y, infinity) {}
  constexpr AffinePoint(Point2<BaseField>&& point, bool infinity = false)
      : AffinePoint(std::move(point.x), std::move(point.y), infinity) {}
  constexpr AffinePoint(const BaseField& x, const BaseField& y,
                        bool infinity = false)
      : x_(x), y_(y), infinity_(infinity) {}
  constexpr AffinePoint(BaseField&& x, BaseField&& y, bool infinity = false)
      : x_(std::move(x)), y_(std::move(y)), infinity_(infinity) {}

  constexpr static AffinePoint CreateChecked(const BaseField& x,
                                             const BaseField& y) {
    AffinePoint ret = {x, y};
    CHECK(ret.IsOnCurve());
    return ret;
  }

  constexpr static AffinePoint CreateChecked(BaseField&& x, BaseField&& y) {
    AffinePoint ret = {std::move(x), std::move(y)};
    CHECK(ret.IsOnCurve());
    return ret;
  }

  constexpr static AffinePoint Zero() { return AffinePoint(); }

  constexpr static AffinePoint Generator() {
    return {Curve::Config::kGenerator.x, Curve::Config::kGenerator.y};
  }

  constexpr static AffinePoint FromProjective(
      const ProjectivePoint<Curve>& point) {
    return point.ToAffine();
  }

  constexpr static AffinePoint FromJacobian(const JacobianPoint<Curve>& point) {
    return point.ToAffine();
  }

  constexpr static AffinePoint FromXYZZ(const PointXYZZ<Curve>& point) {
    return point.ToAffine();
  }

  constexpr static AffinePoint FromMontgomery(
      const Point2<typename BaseField::MontgomeryTy>& point) {
    return {BaseField::FromMontgomery(point.x),
            BaseField::FromMontgomery(point.y)};
  }

  constexpr static AffinePoint Random() {
    return FromJacobian(JacobianPoint<Curve>::Random());
  }

  constexpr static AffinePoint Endomorphism(const AffinePoint& point) {
    return AffinePoint(point.x_ * Curve::Config::kEndomorphismCoefficient,
                       point.y_);
  }

  constexpr const BaseField& x() const { return x_; }
  constexpr const BaseField& y() const { return y_; }
  constexpr bool infinity() const { return infinity_; }

  constexpr bool operator==(const AffinePoint& other) const {
    if (infinity_) {
      return other.infinity_;
    }

    if (other.infinity_) {
      return false;
    }

    return x_ == other.x_ && y_ == other.y_;
  }

  constexpr bool operator!=(const AffinePoint& other) const {
    return !operator==(other);
  }

  constexpr bool IsZero() const { return infinity_; }

  constexpr bool IsOnCurve() { return Curve::IsOnCurve(*this); }

  constexpr ProjectivePoint<Curve> ToProjective() const {
    if (infinity_) return ProjectivePoint<Curve>::Zero();
    return {x_, y_, BaseField::One()};
  }

  constexpr JacobianPoint<Curve> ToJacobian() const {
    if (infinity_) return JacobianPoint<Curve>::Zero();
    return {x_, y_, BaseField::One()};
  }

  constexpr PointXYZZ<Curve> ToXYZZ() const {
    if (infinity_) return PointXYZZ<Curve>::Zero();
    return {x_, y_, BaseField::One(), BaseField::One()};
  }

  constexpr Point2<typename BaseField::MontgomeryTy> ToMontgomery() const {
    return {x_.ToMontgomery(), y_.ToMontgomery()};
  }

  std::string ToString() const {
    return absl::Substitute("($0, $1)", x_.ToString(), y_.ToString());
  }

  std::string ToHexString() const {
    return absl::Substitute("($0, $1)", x_.ToHexString(), y_.ToHexString());
  }

  // AdditiveSemigroup methods
  constexpr JacobianPoint<Curve> Add(const AffinePoint& other) const {
    return ToJacobian() + other.ToJacobian();
  }
  constexpr ProjectivePoint<Curve> Add(
      const ProjectivePoint<Curve>& other) const {
    return ToProjective() + other;
  }
  constexpr JacobianPoint<Curve> Add(const JacobianPoint<Curve>& other) const {
    return ToJacobian() + other;
  }
  constexpr PointXYZZ<Curve> Add(const PointXYZZ<Curve>& other) const {
    return ToXYZZ() + other;
  }

  constexpr AffinePoint& NegInPlace() {
    y_.NegInPlace();
    return *this;
  }

  constexpr ProjectivePoint<Curve> DoubleProjective() const;
  constexpr PointXYZZ<Curve> DoubleXYZZ() const;

  constexpr JacobianPoint<Curve> operator*(const ScalarField& v) const {
    return this->ScalarMul(v.ToBigInt());
  }

 private:
  BaseField x_;
  BaseField y_;
  bool infinity_;
};

}  // namespace tachyon::math

#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point_impl.h"

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_H_
