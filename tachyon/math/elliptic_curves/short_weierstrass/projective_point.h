#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_PROJECTIVE_POINT_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_PROJECTIVE_POINT_H_

#include <type_traits>

#include "absl/strings/substitute.h"

#include "tachyon/base/logging.h"
#include "tachyon/math/base/groups.h"
#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/math/elliptic_curves/curve_config.h"
#include "tachyon/math/elliptic_curves/jacobian_point.h"
#include "tachyon/math/elliptic_curves/point_xyzz.h"
#include "tachyon/math/elliptic_curves/projective_point.h"
#include "tachyon/math/geometry/point3.h"

namespace tachyon::math {

template <typename _Curve>
class ProjectivePoint<_Curve, std::enable_if_t<_Curve::kIsSWCurve>> final
    : public AdditiveGroup<ProjectivePoint<_Curve>> {
 public:
  constexpr static const bool kNegationIsCheap = true;

  using Curve = _Curve;
  using BaseField = typename Curve::BaseField;
  using ScalarField = typename Curve::ScalarField;
  using AffinePointTy = AffinePoint<Curve>;
  using JacobianPointTy = JacobianPoint<Curve>;
  using PointXYZZTy = PointXYZZ<Curve>;

  constexpr ProjectivePoint()
      : ProjectivePoint(BaseField::One(), BaseField::One(), BaseField::Zero()) {
  }
  constexpr ProjectivePoint(const Point3<BaseField>& point)
      : ProjectivePoint(point.x, point.y, point.z) {}
  constexpr ProjectivePoint(Point3<BaseField>&& point)
      : ProjectivePoint(std::move(point.x), std::move(point.y),
                        std::move(point.z)) {}
  constexpr ProjectivePoint(const BaseField& x, const BaseField& y,
                            const BaseField& z)
      : x_(x), y_(y), z_(z) {}
  constexpr ProjectivePoint(BaseField&& x, BaseField&& y, BaseField&& z)
      : x_(std::move(x)), y_(std::move(y)), z_(std::move(z)) {}

  constexpr static ProjectivePoint CreateChecked(const BaseField& x,
                                                 const BaseField& y,
                                                 const BaseField& z) {
    ProjectivePoint ret = {x, y, z};
    CHECK(ret.IsOnCurve());
    return ret;
  }

  constexpr static ProjectivePoint CreateChecked(BaseField&& x, BaseField&& y,
                                                 BaseField&& z) {
    ProjectivePoint ret = {std::move(x), std::move(y), std::move(z)};
    CHECK(ret.IsOnCurve());
    return ret;
  }

  constexpr static ProjectivePoint Zero() { return ProjectivePoint(); }

  constexpr static ProjectivePoint Generator() {
    return {Curve::Config::kGenerator.x, Curve::Config::kGenerator.y,
            BaseField::One()};
  }

  constexpr static ProjectivePoint FromAffine(const AffinePoint<Curve>& point) {
    return point.ToProjective();
  }

  constexpr static ProjectivePoint FromJacobian(
      const JacobianPoint<Curve>& point) {
    return point.ToProjective();
  }

  constexpr static ProjectivePoint FromXYZZ(const PointXYZZ<Curve>& point) {
    return point.ToProjective();
  }

  constexpr static ProjectivePoint FromMontgomery(
      const Point3<typename BaseField::MontgomeryTy>& point) {
    return {BaseField::FromMontgomery(point.x),
            BaseField::FromMontgomery(point.y),
            BaseField::FromMontgomery(point.z)};
  }

  constexpr static ProjectivePoint Random() {
    return ScalarField::Random() * Generator();
  }

  constexpr static ProjectivePoint Endomorphism(const ProjectivePoint& point) {
    return ProjectivePoint(point.x_ * Curve::Config::kEndomorphismCoefficient,
                           point.y_, point.z_);
  }

  constexpr const BaseField& x() const { return x_; }
  constexpr const BaseField& y() const { return y_; }
  constexpr const BaseField& z() const { return z_; }

  constexpr bool operator==(const ProjectivePoint& other) const {
    if (IsZero()) {
      return other.IsZero();
    }

    if (other.IsZero()) {
      return false;
    }

    // The points (X, Y, Z) and (X', Y', Z')
    // are equal when (X * Z') = (X' * Z)
    // and (Y * Z') = (Y' * Z).
    if (x_ * other.z_ != other.x_ * z_) {
      return false;
    } else {
      return y_ * other.z_ == other.y_ * z_;
    }
  }

  constexpr bool operator!=(const ProjectivePoint& other) const {
    return !operator==(other);
  }

  constexpr bool IsZero() const { return z_.IsZero(); }

  constexpr bool IsOnCurve() { return Curve::IsOnCurve(*this); }

  // The jacobian point X, Y, Z is represented in the affine
  // coordinates as X/Z, Y/Z.
  constexpr AffinePoint<Curve> ToAffine() const {
    if (IsZero()) {
      return AffinePoint<Curve>::Zero();
    } else if (z_.IsOne()) {
      return {x_, y_};
    } else {
      BaseField z_inv = z_.Inverse();
      return {x_ * z_inv, y_ * z_inv};
    }
  }

  // The jacobian point X, Y, Z is represented in the jacobian
  // coordinates as X*Z, Y*Z², Z.
  constexpr JacobianPoint<Curve> ToJacobian() const {
    BaseField zz = z_.Square();
    return {x_ * z_, y_ * zz, z_};
  }

  // The jacobian point X, Y, Z is represented in the xyzz
  // coordinates as X*Z, Y*Z², Z², Z³.
  constexpr PointXYZZ<Curve> ToXYZZ() const {
    BaseField zz = z_.Square();
    return {x_ * z_, y_ * zz, zz, z_ * zz};
  }

  constexpr Point3<typename BaseField::MontgomeryTy> ToMontgomery() const {
    return {x_.ToMontgomery(), y_.ToMontgomery(), z_.ToMontgomery()};
  }

  std::string ToString() const {
    return absl::Substitute("($0, $1, $2)", x_.ToString(), y_.ToString(),
                            z_.ToString());
  }

  std::string ToHexString() const {
    return absl::Substitute("($0, $1, $2)", x_.ToHexString(), y_.ToHexString(),
                            z_.ToHexString());
  }

  // AdditiveSemigroup methods
  constexpr ProjectivePoint& AddInPlace(const ProjectivePoint& other);
  constexpr ProjectivePoint& AddInPlace(const AffinePoint<Curve>& other);
  constexpr ProjectivePoint& DoubleInPlace();

  // AdditiveGroup methods
  constexpr ProjectivePoint& NegInPlace() {
    y_.NegInPlace();
    return *this;
  }

  constexpr ProjectivePoint operator*(const ScalarField& v) const {
    return this->ScalarMul(v.ToBigInt());
  }
  constexpr ProjectivePoint& operator*=(const ScalarField& v) {
    return *this = operator*(v);
  }

 private:
  BaseField x_;
  BaseField y_;
  BaseField z_;
};

}  // namespace tachyon::math

#include "tachyon/math/elliptic_curves/short_weierstrass/projective_point_impl.h"

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_PROJECTIVE_POINT_H_
