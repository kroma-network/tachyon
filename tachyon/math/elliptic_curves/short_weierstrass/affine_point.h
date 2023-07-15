#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_H_

#include <type_traits>

#include "absl/strings/substitute.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/math/base/groups.h"
#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/math/elliptic_curves/jacobian_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/sw_curve_config.h"

namespace tachyon {
namespace math {

template <typename Config>
class AffinePoint<Config,
                  std::enable_if_t<std::is_same_v<
                      Config, SWCurveConfig<typename Config::BaseField,
                                            typename Config::ScalarField>>>>
    : public AdditiveGroup<AffinePoint<Config>> {
 public:
  using BaseField = typename Config::BaseField;
  using ScalarField = typename Config::ScalarField;

  constexpr AffinePoint()
      : AffinePoint(BaseField::Zero(), BaseField::Zero(), true) {}
  constexpr AffinePoint(const BaseField& x, const BaseField& y,
                        bool infinity = false)
      : x_(x), y_(y), infinity_(infinity) {}
  constexpr AffinePoint(BaseField&& x, BaseField&& y, bool infinity = false)
      : x_(std::move(x)), y_(std::move(y)), infinity_(infinity) {}

  static AffinePoint CreateChecked(const BaseField& x, const BaseField& y) {
    AffinePoint ret = {x, y};
    CHECK(IsOnCurve(ret));
    return ret;
  }

  static AffinePoint CreateChecked(BaseField&& x, BaseField&& y) {
    AffinePoint ret = {std::move(x), std::move(y)};
    CHECK(IsOnCurve(ret));
    return ret;
  }

  constexpr static AffinePoint Identity() { return AffinePoint(); }

  constexpr static AffinePoint Zero() { return AffinePoint(); }

  constexpr static AffinePoint FromJacobian(
      const JacobianPoint<Config>& point) {
    return point.ToAffine();
  }

  constexpr static AffinePoint Random() {
    return FromJacobian(JacobianPoint<Config>::Random());
  }

  static bool IsOnCurve(const AffinePoint& p) {
    if (p.infinity_) return true;
    return Config::IsOnCurve(p.x_, p.y_);
  }

  template <
      typename BaseInputIterator, typename ScalarInputIterator,
      std::enable_if_t<
          std::is_same_v<AffinePoint, base::iter_value_t<BaseInputIterator>> &&
          std::is_same_v<ScalarField,
                         base::iter_value_t<ScalarInputIterator>>>* = nullptr>
  static JacobianPoint<Config> MSM(BaseInputIterator bases_first,
                                   BaseInputIterator bases_last,
                                   ScalarInputIterator scalars_first,
                                   ScalarInputIterator scalars_last) {
    std::vector<JacobianPoint<Config>> bases =
        base::Map(std::move(bases_first), std::move(bases_last),
                  [](const AffinePoint& point) { return point.ToJacobian(); });
    return Config::MSM(bases.begin(), bases.end(), std::move(scalars_first),
                       std::move(scalars_last));
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

  constexpr JacobianPoint<Config> ToJacobian() const {
    if (infinity_) return JacobianPoint<Config>::Zero();
    return {x_, y_, BaseField::One()};
  }

  std::string ToString() const {
    return absl::Substitute("($0, $1)", x_.ToString(), y_.ToString());
  }

  // AdditiveMonoid methods
  template <typename U>
  constexpr JacobianPoint<Config> Add(const U& other) const {
    JacobianPoint<Config> point = ToJacobian();
    return point + other;
  }

  // AdditiveGroup methods
  template <typename U>
  constexpr JacobianPoint<Config> Sub(const U& other) const {
    JacobianPoint<Config> point = ToJacobian();
    return point - other;
  }

  constexpr AffinePoint& NegInPlace() {
    y_.NegInPlace();
    return *this;
  }

 private:
  BaseField x_;
  BaseField y_;
  bool infinity_;
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_H_
