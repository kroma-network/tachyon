#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_JACOBIAN_POINT_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_JACOBIAN_POINT_H_

#include <type_traits>

#include "absl/strings/substitute.h"

#include "tachyon/base/logging.h"
#include "tachyon/math/base/groups.h"
#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/math/elliptic_curves/jacobian_point.h"
#include "tachyon/math/elliptic_curves/msm/msm_util.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/sw_curve_config.h"

namespace tachyon {
namespace math {

template <typename _Config>
class JacobianPoint<_Config,
                    std::enable_if_t<std::is_same_v<
                        _Config, SWCurveConfig<typename _Config::BaseField,
                                               typename _Config::ScalarField>>>>
    : public AdditiveGroup<JacobianPoint<_Config>> {
 public:
  constexpr static const bool kNegationIsCheap = true;

  using Config = _Config;
  using BaseField = typename Config::BaseField;
  using ScalarField = typename Config::ScalarField;

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
    CHECK(AffinePoint<Config>::IsOnCurve(ret.ToAffine()));
    return ret;
  }

  static JacobianPoint CreateChecked(BaseField&& x, BaseField&& y,
                                     BaseField&& z) {
    JacobianPoint ret = {std::move(x), std::move(y), std::move(z)};
    CHECK(AffinePoint<Config>::IsOnCurve(ret.ToAffine()));
    return ret;
  }

  constexpr static JacobianPoint Zero() { return JacobianPoint(); }

  constexpr static JacobianPoint FromAffine(const AffinePoint<Config>& point) {
    return {point.x(), point.y(), BaseField::One()};
  }

  constexpr static JacobianPoint Random() {
    return ScalarField::Random() * Config::Generator();
  }

  template <
      typename BaseInputIterator, typename ScalarInputIterator,
      std::enable_if_t<IsAbleToMSM<BaseInputIterator, ScalarInputIterator,
                                   JacobianPoint, ScalarField>>* = nullptr>
  static JacobianPoint MSM(BaseInputIterator bases_first,
                           BaseInputIterator bases_last,
                           ScalarInputIterator scalars_first,
                           ScalarInputIterator scalars_last) {
    return Config::MSM(std::move(bases_first), std::move(bases_last),
                       std::move(scalars_first), std::move(scalars_last));
  }

  template <typename BaseContainer, typename ScalarContainer>
  static JacobianPoint<Config> MSM(BaseContainer&& bases,
                                   ScalarContainer&& scalars) {
    return MSM(std::begin(std::forward<BaseContainer>(bases)),
               std::end(std::forward<BaseContainer>(bases)),
               std::begin(std::forward<ScalarContainer>(scalars)),
               std::end(std::forward<ScalarContainer>(scalars)));
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
  constexpr AffinePoint<Config> ToAffine() const {
    if (IsZero()) {
      return AffinePoint<Config>::Identity();
    } else if (z_.IsOne()) {
      return AffinePoint<Config>(x_, y_);
    } else {
      BaseField z_inv = z_.Inverse();
      BaseField z_inv_square = z_inv * z_inv;
      return AffinePoint<Config>(x_ * z_inv_square, y_ * z_inv_square * z_inv);
    }
  }

  std::string ToString() const {
    return absl::Substitute("($0, $1, $2)", x_.ToString(), y_.ToString(),
                            z_.ToString());
  }

  // AdditiveSemigroup methods
  constexpr JacobianPoint& AddInPlace(const JacobianPoint& other);
  constexpr JacobianPoint& AddInPlace(const AffinePoint<Config>& other);
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
