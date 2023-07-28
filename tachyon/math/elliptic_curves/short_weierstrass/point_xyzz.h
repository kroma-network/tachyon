#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_POINT_XYZZ_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_POINT_XYZZ_H_

#include <type_traits>

#include "absl/strings/substitute.h"

#include "tachyon/base/logging.h"
#include "tachyon/math/base/groups.h"
#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/math/elliptic_curves/curve_config.h"
#include "tachyon/math/elliptic_curves/msm/glv.h"
#include "tachyon/math/elliptic_curves/msm/msm_util.h"
#include "tachyon/math/elliptic_curves/point_xyzz.h"
#include "tachyon/math/elliptic_curves/projective_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/sw_curve.h"
#include "tachyon/math/geometry/point4.h"

namespace tachyon::math {

template <typename _Curve>
class PointXYZZ<_Curve, std::enable_if_t<_Curve::kIsSWCurve>>
    : public AdditiveGroup<PointXYZZ<_Curve>> {
 public:
  constexpr static const bool kNegationIsCheap = true;

  using Curve = _Curve;
  using BaseField = typename Curve::BaseField;
  using ScalarField = typename Curve::ScalarField;
  using AffinePointTy = AffinePoint<Curve>;
  using ProjectivePointTy = ProjectivePoint<Curve>;
  using JacobianPointTy = JacobianPoint<Curve>;

  constexpr PointXYZZ()
      : PointXYZZ(BaseField::One(), BaseField::One(), BaseField::Zero(),
                  BaseField::Zero()) {}
  constexpr PointXYZZ(const BaseField& x, const BaseField& y,
                      const BaseField& zz, const BaseField& zzz)
      : x_(x), y_(y), zz_(zz), zzz_(zzz) {}
  constexpr PointXYZZ(BaseField&& x, BaseField&& y, BaseField&& zz,
                      BaseField&& zzz)
      : x_(std::move(x)),
        y_(std::move(y)),
        zz_(std::move(zz)),
        zzz_(std::move(zzz)) {}

  constexpr static PointXYZZ CreateChecked(const BaseField& x,
                                           const BaseField& y,
                                           const BaseField& zz,
                                           const BaseField& zzz) {
    PointXYZZ ret = {x, y, zz, zzz};
    CHECK(Curve::IsOnCurve(ret));
    return ret;
  }

  constexpr static PointXYZZ CreateChecked(BaseField&& x, BaseField&& y,
                                           BaseField&& zz, BaseField&& zzz) {
    PointXYZZ ret = {std::move(x), std::move(y), std::move(zz), std::move(zzz)};
    CHECK(Curve::IsOnCurve(ret));
    return ret;
  }

  constexpr static PointXYZZ Zero() { return PointXYZZ(); }

  constexpr static PointXYZZ FromAffine(const AffinePoint<Curve>& point) {
    return point.ToXYZZ();
  }

  constexpr static PointXYZZ FromProjective(
      const ProjectivePoint<Curve>& point) {
    return point.ToXYZZ();
  }

  constexpr static PointXYZZ FromJacobian(const JacobianPoint<Curve>& point) {
    return point.ToXYZZ();
  }

  constexpr static PointXYZZ FromMontgomery(
      const Point4<typename BaseField::BigIntTy>& point) {
    return {
        BaseField::FromMontgomery(point.x), BaseField::FromMontgomery(point.y),
        BaseField::FromMontgomery(point.z), BaseField::FromMontgomery(point.w)};
  }

  constexpr static PointXYZZ Random() {
    return FromJacobian(JacobianPoint<Curve>::Random());
  }

  constexpr static bool IsOnCurve(const PointXYZZ& p) {
    return Curve::IsOnCurve(p);
  }

  template <typename BaseInputIterator, typename ScalarInputIterator,
            std::enable_if_t<IsAbleToMSM<BaseInputIterator, ScalarInputIterator,
                                         PointXYZZ, ScalarField>>* = nullptr>
  static PointXYZZ MSM(BaseInputIterator bases_first,
                       BaseInputIterator bases_last,
                       ScalarInputIterator scalars_first,
                       ScalarInputIterator scalars_last) {
    return Curve::template MSM<PointXYZZ>(
        std::move(bases_first), std::move(bases_last), std::move(scalars_first),
        std::move(scalars_last));
  }

  template <typename BaseContainer, typename ScalarContainer>
  static PointXYZZ MSM(BaseContainer&& bases, ScalarContainer&& scalars) {
    return MSM(std::begin(std::forward<BaseContainer>(bases)),
               std::end(std::forward<BaseContainer>(bases)),
               std::begin(std::forward<ScalarContainer>(scalars)),
               std::end(std::forward<ScalarContainer>(scalars)));
  }

  constexpr static PointXYZZ Endomorphism(const PointXYZZ& point) {
    return PointXYZZ(point.x_ * GLV<PointXYZZ>::EndomorphismCoefficient(),
                     point.y_, point.zz_, point.zzz_);
  }

  constexpr const BaseField& x() const { return x_; }
  constexpr const BaseField& y() const { return y_; }
  constexpr const BaseField& zz() const { return zz_; }
  constexpr const BaseField& zzz() const { return zzz_; }

  constexpr bool operator==(const PointXYZZ& other) const {
    if (IsZero()) {
      return other.IsZero();
    }

    if (other.IsZero()) {
      return false;
    }

    // The points (X, Y, ZZ, ZZZ) and (X', Y', ZZ', ZZZ')
    // are equal when (X * ZZ') = (X' * ZZ)
    // and (Y * Z'³) = (Y' * Z³).
    if (x_ * other.zz_ != other.x_ * zz_) {
      return false;
    } else {
      return y_ * other.zzz_ == other.y_ * zzz_;
    }
  }

  constexpr bool operator!=(const PointXYZZ& other) const {
    return !operator==(other);
  }

  constexpr bool IsZero() const { return zz_.IsZero(); }

  // The xyzz point X, Y, ZZ, ZZZ is represented in the affine
  // coordinates as X/ZZ, Y/ZZZ.
  constexpr AffinePoint<Curve> ToAffine() const {
    if (IsZero()) {
      return AffinePoint<Curve>::Zero();
    } else if (zz_.IsOne()) {
      return {x_, y_};
    } else {
      BaseField z_inv_cubic = zzz_.Inverse();
      BaseField z_inv_square = z_inv_cubic * zz_;
      z_inv_square.SquareInPlace();
      return {x_ * z_inv_square, y_ * z_inv_cubic};
    }
  }

  // The xyzz point X, Y, ZZ, ZZZ is represented in the projective
  // coordinates as X*ZZZ, Y*ZZ, ZZ*ZZZ.
  constexpr ProjectivePoint<Curve> ToProjective() const {
    if (IsZero()) {
      return ProjectivePoint<Curve>::Zero();
    } else if (zz_.IsOne()) {
      return {x_, y_, BaseField::One()};
    } else {
      return {x_ * zzz_, y_ * zz_, zz_ * zzz_};
    }
  }

  // The xyzz point X, Y, ZZ, ZZZ is represented in the jacobian
  // coordinates as X*ZZZ²*ZZ, Y*ZZ³*ZZZ², ZZZ*ZZ.
  constexpr JacobianPoint<Curve> ToJacobian() const {
    if (IsZero()) {
      return JacobianPoint<Curve>::Zero();
    } else if (zz_.IsOne()) {
      return {x_, y_, BaseField::One()};
    } else {
      BaseField z = zz_ * zzz_;
      return {x_ * zzz_ * z, y_ * zz_ * z.Square(), z};
    }
  }

  constexpr Point4<typename BaseField::BigIntTy> ToMontgomery() const {
    return {x_.ToMontgomery(), y_.ToMontgomery(), zz_.ToMontgomery(),
            zzz_.ToMontgomery()};
  }

  std::string ToString() const {
    return absl::Substitute("($0, $1, $2, $3)", x_.ToString(), y_.ToString(),
                            zz_.ToString(), zzz_.ToString());
  }

  // AdditiveSemigroup methods
  constexpr PointXYZZ& AddInPlace(const PointXYZZ& other);
  constexpr PointXYZZ& AddInPlace(const AffinePoint<Curve>& other);
  constexpr PointXYZZ& DoubleInPlace();

  // AdditiveGroup methods
  constexpr PointXYZZ& NegInPlace() {
    y_.NegInPlace();
    return *this;
  }

 private:
  BaseField x_;
  BaseField y_;
  BaseField zz_;
  BaseField zzz_;
};

}  // namespace tachyon::math

#include "tachyon/math/elliptic_curves/short_weierstrass/point_xyzz_impl.h"

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_POINT_XYZZ_H_
