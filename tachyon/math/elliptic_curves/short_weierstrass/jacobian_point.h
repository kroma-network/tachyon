#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_JACOBIAN_POINT_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_JACOBIAN_POINT_H_

#include <optional>
#include <string>
#include <type_traits>
#include <utility>

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
class JacobianPoint<_Curve, std::enable_if_t<_Curve::kIsSWCurve>> final
    : public AdditiveGroup<JacobianPoint<_Curve>> {
 public:
  constexpr static bool kNegationIsCheap = true;

  using Curve = _Curve;
  using BaseField = typename Curve::BaseField;
  using ScalarField = typename Curve::ScalarField;
  using AffinePointTy = AffinePoint<Curve>;
  using ProjectivePointTy = ProjectivePoint<Curve>;
  using PointXYZZTy = PointXYZZ<Curve>;

  constexpr JacobianPoint()
      : JacobianPoint(BaseField::One(), BaseField::One(), BaseField::Zero()) {}
  explicit constexpr JacobianPoint(const Point3<BaseField>& point)
      : JacobianPoint(point.x, point.y, point.z) {}
  explicit constexpr JacobianPoint(Point3<BaseField>&& point)
      : JacobianPoint(std::move(point.x), std::move(point.y),
                      std::move(point.z)) {}
  constexpr JacobianPoint(const BaseField& x, const BaseField& y,
                          const BaseField& z)
      : x_(x), y_(y), z_(z) {}
  constexpr JacobianPoint(BaseField&& x, BaseField&& y, BaseField&& z)
      : x_(std::move(x)), y_(std::move(y)), z_(std::move(z)) {}

  constexpr static JacobianPoint CreateChecked(const BaseField& x,
                                               const BaseField& y,
                                               const BaseField& z) {
    JacobianPoint ret = {x, y, z};
    CHECK(ret.IsOnCurve());
    return ret;
  }

  constexpr static JacobianPoint CreateChecked(BaseField&& x, BaseField&& y,
                                               BaseField&& z) {
    JacobianPoint ret = {std::move(x), std::move(y), std::move(z)};
    CHECK(ret.IsOnCurve());
    return ret;
  }

  constexpr static std::optional<JacobianPoint> CreateFromX(const BaseField& x,
                                                            bool pick_odd) {
    JacobianPoint point;
    if (!Curve::GetPointFromX(x, pick_odd, &point)) return std::nullopt;
    return point;
  }

  constexpr static JacobianPoint Zero() { return JacobianPoint(); }

  constexpr static JacobianPoint Generator() {
    return {Curve::Config::kGenerator.x, Curve::Config::kGenerator.y,
            BaseField::One()};
  }

  constexpr static JacobianPoint FromAffine(const AffinePoint<Curve>& point) {
    return point.ToJacobian();
  }

  constexpr static JacobianPoint FromProjective(
      const ProjectivePoint<Curve>& point) {
    return point.ToJacobian();
  }

  constexpr static JacobianPoint FromXYZZ(const PointXYZZ<Curve>& point) {
    return point.ToJacobian();
  }

  constexpr static JacobianPoint FromMontgomery(
      const Point3<typename BaseField::MontgomeryTy>& point) {
    return {BaseField::FromMontgomery(point.x),
            BaseField::FromMontgomery(point.y),
            BaseField::FromMontgomery(point.z)};
  }

  constexpr static JacobianPoint Random() {
    return ScalarField::Random() * Generator();
  }

  constexpr static JacobianPoint Endomorphism(const JacobianPoint& point) {
    return JacobianPoint(point.x_ * Curve::Config::kEndomorphismCoefficient,
                         point.y_, point.z_);
  }

  constexpr const BaseField& x() const { return x_; }
  constexpr const BaseField& y() const { return y_; }
  constexpr const BaseField& z() const { return z_; }

  constexpr bool operator==(const JacobianPoint& other) const {
    if (IsZero()) {
      return other.IsZero();
    }

    if (other.IsZero()) {
      return false;
    }

    // The points (X, Y, Z) and (X', Y', Z')
    // are equal when (X * Z'²) = (X' * Z²)
    // and (Y * Z'³) = (Y' * Z³).
    const BaseField z1z1 = z_ * z_;
    const BaseField z2z2 = other.z_ * other.z_;

    if (x_ * z2z2 != other.x_ * z1z1) {
      return false;
    } else {
      return y_ * (z2z2 * other.z_) == other.y_ * (z1z1 * z_);
    }
  }

  constexpr bool operator!=(const JacobianPoint& other) const {
    return !operator==(other);
  }

  constexpr bool IsZero() const { return z_.IsZero(); }

  constexpr bool IsOnCurve() { return Curve::IsOnCurve(*this); }

  // The jacobian point X, Y, Z is represented in the affine
  // coordinates as X/Z², Y/Z³.
  constexpr AffinePoint<Curve> ToAffine() const {
    if (IsZero()) {
      return AffinePoint<Curve>::Zero();
    } else if (z_.IsOne()) {
      return {x_, y_};
    } else {
      BaseField z_inv = z_.Inverse();
      BaseField z_inv_square = z_inv * z_inv;
      return {x_ * z_inv_square, y_ * z_inv_square * z_inv};
    }
  }

  // The jacobian point X, Y, Z is represented in the projective
  // coordinates as X*Z, Y, Z³.
  constexpr ProjectivePoint<Curve> ToProjective() const {
    BaseField zz = z_.Square();
    return {x_ * z_, y_, zz * z_};
  }

  // The jacobian point X, Y, Z is represented in the xyzz
  // coordinates as X, Y, Z², Z³.
  constexpr PointXYZZ<Curve> ToXYZZ() const {
    BaseField zz = z_.Square();
    return {x_, y_, zz, zz * z_};
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
  constexpr JacobianPoint& AddInPlace(const JacobianPoint& other);
  constexpr JacobianPoint& AddInPlace(const AffinePoint<Curve>& other);
  constexpr JacobianPoint& DoubleInPlace();

  // AdditiveGroup methods
  constexpr JacobianPoint& NegInPlace() {
    y_.NegInPlace();
    return *this;
  }

  constexpr JacobianPoint operator*(const ScalarField& v) const {
    return this->ScalarMul(v.ToBigInt());
  }
  constexpr JacobianPoint& operator*=(const ScalarField& v) {
    return *this = operator*(v);
  }

 private:
  BaseField x_;
  BaseField y_;
  BaseField z_;
};

}  // namespace tachyon::math

#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point_impl.h"

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_JACOBIAN_POINT_H_
