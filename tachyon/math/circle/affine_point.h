// Copyright 2024 StarkWare Industries Ltd
// Use of this source code is governed by a Apache-2.0 style license that
// can be found in the LICENSE-APACHE.stwo

#ifndef TACHYON_MATH_CIRCLE_AFFINE_POINT_H_
#define TACHYON_MATH_CIRCLE_AFFINE_POINT_H_

#include <string>
#include <utility>

#include "absl/strings/substitute.h"

#include "tachyon/base/logging.h"
#include "tachyon/math/base/groups.h"
#include "tachyon/math/geometry/affine_point.h"
#include "tachyon/math/geometry/curve_type.h"
#include "tachyon/math/geometry/point2.h"

namespace tachyon::math {

template <typename _Curve>
class AffinePoint<_Curve, std::enable_if_t<_Curve::kType == CurveType::kCircle>>
    final : public AdditiveGroup<AffinePoint<_Curve>> {
 public:
  using Curve = _Curve;
  using Circle = _Curve;
  using BaseField = typename Circle::BaseField;
  using ScalarField = typename Circle::ScalarField;

  constexpr AffinePoint() : AffinePoint(BaseField::One(), BaseField::Zero()) {}
  explicit constexpr AffinePoint(const Point2<BaseField>& point)
      : AffinePoint(point.x, point.y) {}
  explicit constexpr AffinePoint(Point2<BaseField>&& point)
      : AffinePoint(std::move(point.x), std::move(point.y)) {}
  constexpr AffinePoint(const BaseField& x, const BaseField& y)
      : x_(x), y_(y) {}
  constexpr AffinePoint(BaseField&& x, BaseField&& y)
      : x_(std::move(x)), y_(std::move(y)) {}

  constexpr static AffinePoint CreateChecked(const BaseField& x,
                                             const BaseField& y) {
    AffinePoint ret = {x, y};
    CHECK(ret.IsOnCircle());
    return ret;
  }

  constexpr static AffinePoint CreateChecked(BaseField&& x, BaseField&& y) {
    AffinePoint ret = {std::move(x), std::move(y)};
    CHECK(ret.IsOnCircle());
    return ret;
  }

  constexpr static AffinePoint Zero() { return AffinePoint(); }

  constexpr static AffinePoint One() { return Generator(); }

  constexpr static AffinePoint Generator() {
    return {Circle::Config::kGenerator.x, Circle::Config::kGenerator.y};
  }

  constexpr static AffinePoint Random() {
    return ScalarField::Random() * Generator();
  }

  constexpr const BaseField& x() const { return x_; }
  constexpr const BaseField& y() const { return y_; }

  constexpr bool operator==(const AffinePoint& other) const {
    return x_ == other.x_ && y_ == other.y_;
  }
  constexpr bool operator!=(const AffinePoint& other) const {
    return !operator==(other);
  }

  constexpr bool IsZero() const { return x_.IsOne() && y_.IsZero(); }

  constexpr bool IsOnCircle() { return Circle::IsOnCircle(*this); }

  constexpr AffinePoint Conjugate() const { return {x_, -y_}; }

  constexpr AffinePoint ComplexConjugate() const {
    return {x_.ComplexConjugate(), y_.ComplexConjugate()};
  }

  constexpr AffinePoint Antipode() const { return {-x_, -y_}; }

  std::string ToString() const {
    return absl::Substitute("($0, $1)", x_.ToString(), y_.ToString());
  }

  std::string ToHexString(bool pad_zero = false) const {
    return absl::Substitute("($0, $1)", x_.ToHexString(pad_zero),
                            y_.ToHexString(pad_zero));
  }

  // AdditiveSemigroup methods
  constexpr AffinePoint Add(const AffinePoint& other) const {
    return {x_ * other.x_ - y_ * other.y_, x_ * other.y_ + y_ * other.x_};
  }

  constexpr AffinePoint& AddInPlace(const AffinePoint& other) {
    return *this = {x_ * other.x_ - y_ * other.y_,
                    x_ * other.y_ + y_ * other.x_};
  }

  constexpr AffinePoint DoubleImpl() const {
    return {x_.Square().Double() - BaseField::One(), x_.Double() * y_};
  }

  constexpr AffinePoint& DoubleImplInPlace() {
    y_ *= x_.Double();
    x_.SquareInPlace().DoubleInPlace() -= BaseField::One();
    return *this;
  }

  // AdditiveGroup methods
  constexpr AffinePoint Sub(const AffinePoint& other) const {
    return {x_ * other.x_ + y_ * other.y_, -x_ * other.y_ + y_ * other.x_};
  }

  constexpr AffinePoint& SubInPlace(const AffinePoint& other) {
    return *this = {x_ * other.x_ + y_ * other.y_,
                    -x_ * other.y_ + y_ * other.x_};
  }

  constexpr AffinePoint Negate() const { return {x_, -y_}; }

  constexpr AffinePoint& NegateInPlace() {
    y_.NegateInPlace();
    return *this;
  }

  constexpr AffinePoint operator*(const ScalarField& idx) const {
    return this->ScalarMul(idx.value());
  }

 private:
  BaseField x_;
  BaseField y_;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_CIRCLE_AFFINE_POINT_H_
