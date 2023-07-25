#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_H_

#include <type_traits>

#include "absl/strings/substitute.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/math/base/groups.h"
#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/math/elliptic_curves/curve_config.h"
#include "tachyon/math/elliptic_curves/jacobian_point.h"
#include "tachyon/math/elliptic_curves/msm/glv.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/sw_curve.h"

namespace tachyon {
namespace math {

template <typename _Curve>
class AffinePoint<_Curve, std::enable_if_t<_Curve::kIsSWCurve>>
    : public AdditiveGroup<AffinePoint<_Curve>> {
 public:
  using Curve = _Curve;
  using BaseField = typename Curve::BaseField;
  using ScalarField = typename Curve::ScalarField;
  using JacobianPointTy = JacobianPoint<Curve>;

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

  constexpr static AffinePoint FromJacobian(const JacobianPoint<Curve>& point) {
    return point.ToAffine();
  }

  constexpr static AffinePoint Random() {
    return FromJacobian(JacobianPoint<Curve>::Random());
  }

  static bool IsOnCurve(const AffinePoint& p) {
    if (p.infinity_) return true;
    return Curve::IsOnCurve(p.x_, p.y_);
  }

  template <
      typename BaseInputIterator, typename ScalarInputIterator,
      std::enable_if_t<
          std::is_same_v<AffinePoint, base::iter_value_t<BaseInputIterator>> &&
          std::is_same_v<ScalarField,
                         base::iter_value_t<ScalarInputIterator>>>* = nullptr>
  static JacobianPoint<Curve> MSM(BaseInputIterator bases_first,
                                  BaseInputIterator bases_last,
                                  ScalarInputIterator scalars_first,
                                  ScalarInputIterator scalars_last) {
    std::vector<JacobianPoint<Curve>> bases =
        base::Map(std::move(bases_first), std::move(bases_last),
                  [](const AffinePoint& point) { return point.ToJacobian(); });
    return Curve::MSM(bases.begin(), bases.end(), std::move(scalars_first),
                      std::move(scalars_last));
  }

  template <typename BaseContainer, typename ScalarContainer>
  static JacobianPoint<Curve> MSM(BaseContainer&& bases,
                                  ScalarContainer&& scalars) {
    return MSM(std::begin(std::forward<BaseContainer>(bases)),
               std::end(std::forward<BaseContainer>(bases)),
               std::begin(std::forward<ScalarContainer>(scalars)),
               std::end(std::forward<ScalarContainer>(scalars)));
  }

  constexpr static AffinePoint EndomorphismAffine(const AffinePoint& point) {
    return AffinePoint(point.x_ * GLV<Curve>::EndomorphismCoefficient(),
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

  constexpr JacobianPoint<Curve> ToJacobian() const {
    if (infinity_) return JacobianPoint<Curve>::Zero();
    return {x_, y_, BaseField::One()};
  }

  std::string ToString() const {
    return absl::Substitute("($0, $1)", x_.ToString(), y_.ToString());
  }

  // AdditiveSemigroup methods
  template <typename U>
  constexpr JacobianPoint<Curve> Add(const U& other) const {
    JacobianPoint<Curve> point = ToJacobian();
    return point + other;
  }

  // AdditiveGroup methods
  template <typename U>
  constexpr JacobianPoint<Curve> Sub(const U& other) const {
    JacobianPoint<Curve> point = ToJacobian();
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
