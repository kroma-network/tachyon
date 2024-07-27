// Copyright 2024 StarkWare Industries Ltd
// Use of this source code is governed by a Apache-2.0 style license that
// can be found in the LICENSE-APACHE.stwo

#ifndef TACHYON_MATH_CIRCLE_CIRCLE_POINT_H_
#define TACHYON_MATH_CIRCLE_CIRCLE_POINT_H_

#include <string>
#include <utility>

#include "absl/strings/substitute.h"

#include "tachyon/base/logging.h"
#include "tachyon/math/base/groups.h"
#include "tachyon/math/circle/circle_point_forward.h"
#include "tachyon/math/geometry/point2.h"

namespace tachyon::math {

// TODO(chokobole): Unify this into |AffinePoint|.
template <typename _Circle>
class CirclePoint : public AdditiveGroup<CirclePoint<_Circle>> {
 public:
  using Circle = _Circle;
  using BaseField = typename Circle::BaseField;
  using ScalarField = typename Circle::ScalarField;

  constexpr CirclePoint() : CirclePoint(BaseField::One(), BaseField::Zero()) {}
  explicit constexpr CirclePoint(const Point2<BaseField>& point)
      : CirclePoint(point.x, point.y) {}
  explicit constexpr CirclePoint(Point2<BaseField>&& point)
      : CirclePoint(std::move(point.x), std::move(point.y)) {}
  constexpr CirclePoint(const BaseField& x, const BaseField& y)
      : x_(x), y_(y) {}
  constexpr CirclePoint(BaseField&& x, BaseField&& y)
      : x_(std::move(x)), y_(std::move(y)) {}

  constexpr static CirclePoint CreateChecked(const BaseField& x,
                                             const BaseField& y) {
    CirclePoint ret = {x, y};
    CHECK(ret.IsOnCircle());
    return ret;
  }

  constexpr static CirclePoint CreateChecked(BaseField&& x, BaseField&& y) {
    CirclePoint ret = {std::move(x), std::move(y)};
    CHECK(ret.IsOnCircle());
    return ret;
  }

  constexpr static CirclePoint Zero() { return CirclePoint(); }

  constexpr static CirclePoint One() { return Generator(); }

  constexpr static CirclePoint Generator() {
    return {Circle::Config::kGenerator.x, Circle::Config::kGenerator.y};
  }

  constexpr static CirclePoint Random() {
    return ScalarField::Random() * Generator();
  }

  constexpr const BaseField& x() const { return x_; }
  constexpr const BaseField& y() const { return y_; }

  constexpr bool operator==(const CirclePoint& other) const {
    return x_ == other.x_ && y_ == other.y_;
  }
  constexpr bool operator!=(const CirclePoint& other) const {
    return !operator==(other);
  }

  constexpr bool IsZero() const { return x_.IsOne() && y_.IsZero(); }

  constexpr bool IsOnCircle() { return Circle::IsOnCircle(*this); }

  constexpr CirclePoint Conjugate() const { return {x_, -y_}; }

  constexpr CirclePoint ComplexConjugate() const {
    return {x_.ComplexConjugate(), y_.ComplexConjugate()};
  }

  constexpr CirclePoint Antipode() const { return {-x_, -y_}; }

  std::string ToString() const {
    return absl::Substitute("($0, $1)", x_.ToString(), y_.ToString());
  }

  std::string ToHexString(bool pad_zero = false) const {
    return absl::Substitute("($0, $1)", x_.ToHexString(pad_zero),
                            y_.ToHexString(pad_zero));
  }

  // AdditiveSemigroup methods
  constexpr CirclePoint Add(const CirclePoint& other) const {
    return {x_ * other.x_ - y_ * other.y_, x_ * other.y_ + y_ * other.x_};
  }

  constexpr CirclePoint& AddInPlace(const CirclePoint& other) {
    return *this = {x_ * other.x_ - y_ * other.y_,
                    x_ * other.y_ + y_ * other.x_};
  }

  constexpr CirclePoint DoubleImpl() const {
    return {x_.Square().Double() - BaseField::One(), x_.Double() * y_};
  }

  constexpr CirclePoint& DoubleImplInPlace() {
    y_ *= x_.Double();
    x_.SquareInPlace().DoubleInPlace() -= BaseField::One();
    return *this;
  }

  // AdditiveGroup methods
  constexpr CirclePoint Sub(const CirclePoint& other) const {
    return {x_ * other.x_ + y_ * other.y_, -x_ * other.y_ + y_ * other.x_};
  }

  constexpr CirclePoint& SubInPlace(const CirclePoint& other) {
    return *this = {x_ * other.x_ + y_ * other.y_,
                    -x_ * other.y_ + y_ * other.x_};
  }

  constexpr CirclePoint Negate() const { return {x_, -y_}; }

  constexpr CirclePoint& NegateInPlace() {
    y_.NegateInPlace();
    return *this;
  }

  constexpr CirclePoint operator*(const ScalarField& idx) const {
    return this->ScalarMul(idx.value());
  }

 private:
  BaseField x_;
  BaseField y_;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_CIRCLE_CIRCLE_POINT_H_
