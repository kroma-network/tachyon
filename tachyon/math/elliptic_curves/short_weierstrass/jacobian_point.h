#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_JACOBIAN_POINT_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_JACOBIAN_POINT_H_

#include <type_traits>

#include "absl/strings/substitute.h"

#include "tachyon/base/logging.h"
#include "tachyon/math/base/groups.h"
#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/math/elliptic_curves/curve_config.h"
#include "tachyon/math/elliptic_curves/jacobian_point.h"
#include "tachyon/math/elliptic_curves/msm/glv.h"
#include "tachyon/math/elliptic_curves/msm/msm_util.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/sw_curve.h"
#include "tachyon/math/geometry/point3.h"

namespace tachyon {
namespace math {

template <typename _Curve>
class JacobianPoint<_Curve, std::enable_if_t<_Curve::kIsSWCurve>>
    : public AdditiveGroup<JacobianPoint<_Curve>> {
 public:
  constexpr static const bool kNegationIsCheap = true;

  using Curve = _Curve;
  using BaseField = typename Curve::BaseField;
  using ScalarField = typename Curve::ScalarField;
  using AffinePointTy = AffinePoint<Curve>;

  constexpr JacobianPoint()
      : JacobianPoint(BaseField::One(), BaseField::One(), BaseField::Zero()) {}
  constexpr JacobianPoint(const BaseField& x, const BaseField& y,
                          const BaseField& z)
      : x_(x), y_(y), z_(z) {}
  constexpr JacobianPoint(BaseField&& x, BaseField&& y, BaseField&& z)
      : x_(std::move(x)), y_(std::move(y)), z_(std::move(z)) {}

  static JacobianPoint CreateChecked(const BaseField& x, const BaseField& y,
                                     const BaseField& z) {
    JacobianPoint ret = {x, y, z};
    CHECK(AffinePoint<Curve>::IsOnCurve(ret.ToAffine()));
    return ret;
  }

  static JacobianPoint CreateChecked(BaseField&& x, BaseField&& y,
                                     BaseField&& z) {
    JacobianPoint ret = {std::move(x), std::move(y), std::move(z)};
    CHECK(AffinePoint<Curve>::IsOnCurve(ret.ToAffine()));
    return ret;
  }

  constexpr static JacobianPoint Zero() { return JacobianPoint(); }

  constexpr static JacobianPoint FromAffine(const AffinePoint<Curve>& point) {
    return {point.x(), point.y(), BaseField::One()};
  }

  constexpr static JacobianPoint FromMontgomery(
      const Point3<typename BaseField::BigIntTy>& point) {
    return JacobianPoint(BaseField::FromMontgomery(point.x),
                         BaseField::FromMontgomery(point.y),
                         BaseField::FromMontgomery(point.z));
  }

  constexpr static JacobianPoint Random() {
    return ScalarField::Random() * Curve::Generator();
  }

  template <
      typename BaseInputIterator, typename ScalarInputIterator,
      std::enable_if_t<IsAbleToMSM<BaseInputIterator, ScalarInputIterator,
                                   JacobianPoint, ScalarField>>* = nullptr>
  static JacobianPoint MSM(BaseInputIterator bases_first,
                           BaseInputIterator bases_last,
                           ScalarInputIterator scalars_first,
                           ScalarInputIterator scalars_last) {
    return Curve::MSM(std::move(bases_first), std::move(bases_last),
                      std::move(scalars_first), std::move(scalars_last));
  }

  template <typename BaseContainer, typename ScalarContainer>
  static JacobianPoint MSM(BaseContainer&& bases, ScalarContainer&& scalars) {
    return MSM(std::begin(std::forward<BaseContainer>(bases)),
               std::end(std::forward<BaseContainer>(bases)),
               std::begin(std::forward<ScalarContainer>(scalars)),
               std::end(std::forward<ScalarContainer>(scalars)));
  }

  constexpr static JacobianPoint Endomorphism(const JacobianPoint& point) {
    return JacobianPoint(point.x_ * GLV<Curve>::EndomorphismCoefficient(),
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
    // are equal when (X * Z²) = (X' * Z'²)
    // and (Y * Z³) = (Y' * Z'³).
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

  // The jacobian point X, Y, Z is represented in the affine
  // coordinates as X/Z², Y/Z³.
  constexpr AffinePoint<Curve> ToAffine() const {
    if (IsZero()) {
      return AffinePoint<Curve>::Identity();
    } else if (z_.IsOne()) {
      return AffinePoint<Curve>(x_, y_);
    } else {
      BaseField z_inv = z_.Inverse();
      BaseField z_inv_square = z_inv * z_inv;
      return AffinePoint<Curve>(x_ * z_inv_square, y_ * z_inv_square * z_inv);
    }
  }

  constexpr Point3<typename BaseField::BigIntTy> ToMontgomery() const {
    return {x_.ToMontgomery(), y_.ToMontgomery(), z_.ToMontgomery()};
  }

  std::string ToString() const {
    return absl::Substitute("($0, $1, $2)", x_.ToString(), y_.ToString(),
                            z_.ToString());
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

 private:
  BaseField x_;
  BaseField y_;
  BaseField z_;
};

}  // namespace math
}  // namespace tachyon

#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point_impl.h"

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_PROJECTIVE_POINT_H_
