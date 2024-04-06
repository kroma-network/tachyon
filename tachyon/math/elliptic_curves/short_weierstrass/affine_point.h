#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_H_

#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/json/json.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/parallelize.h"
#include "tachyon/math/base/groups.h"
#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/math/elliptic_curves/curve_type.h"
#include "tachyon/math/elliptic_curves/jacobian_point.h"
#include "tachyon/math/elliptic_curves/point_xyzz.h"
#include "tachyon/math/elliptic_curves/projective_point.h"
#include "tachyon/math/elliptic_curves/semigroups.h"
#include "tachyon/math/geometry/point2.h"

namespace tachyon {
namespace math {

template <typename _Curve>
class AffinePoint<
    _Curve, std::enable_if_t<_Curve::kType == CurveType::kShortWeierstrass>>
    final : public AdditiveGroup<AffinePoint<_Curve>> {
 public:
  constexpr static bool kNegationIsCheap = true;

  using Curve = _Curve;
  using BaseField = typename Curve::BaseField;
  using ScalarField = typename Curve::ScalarField;

  constexpr AffinePoint()
      : AffinePoint(BaseField::Zero(), BaseField::Zero(), true) {}
  explicit constexpr AffinePoint(const Point2<BaseField>& point)
      : AffinePoint(point.x, point.y, false) {}
  constexpr AffinePoint(const Point2<BaseField>& point, bool infinity)
      : AffinePoint(point.x, point.y, infinity) {}
  explicit constexpr AffinePoint(Point2<BaseField>&& point)
      : AffinePoint(std::move(point.x), std::move(point.y), false) {}
  constexpr AffinePoint(Point2<BaseField>&& point, bool infinity)
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

  constexpr static std::optional<AffinePoint> CreateFromX(const BaseField& x,
                                                          bool pick_odd) {
    AffinePoint point;
    if (!Curve::GetPointFromX(x, pick_odd, &point)) return std::nullopt;
    return point;
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

  template <typename ScalarFieldContainer, typename AffineContainer>
  [[nodiscard]] constexpr static bool BatchMapScalarFieldToPoint(
      const AffinePoint& point, const ScalarFieldContainer& scalar_fields,
      AffineContainer* affine_points) {
    return DoBatchMapScalarFieldToPoint(point, scalar_fields, affine_points);
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

  std::string ToHexString(bool pad_zero = false) const {
    return absl::Substitute("($0, $1)", x_.ToHexString(pad_zero),
                            y_.ToHexString(pad_zero));
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

  constexpr AffinePoint Negative() const { return {x_, -y_, infinity_}; }

  constexpr AffinePoint& NegInPlace() {
    y_.NegInPlace();
    return *this;
  }

  constexpr ProjectivePoint<Curve> DoubleProjective() const;
  constexpr PointXYZZ<Curve> DoubleXYZZ() const;

  constexpr JacobianPoint<Curve> operator*(const ScalarField& v) const {
    return this->ScalarMul(v);
  }

 private:
  template <typename ScalarFieldContainer, typename AffineContainer>
  [[nodiscard]] constexpr static bool DoBatchMapScalarFieldToPoint(
      const AffinePoint& point, const ScalarFieldContainer& scalar_fields,
      AffineContainer* affine_points) {
    size_t size = std::size(scalar_fields);
    if (size != std::size(*affine_points)) {
      LOG(ERROR) << "Size of |scalar_fields| and |affine_points| do not match";
      return false;
    }
    std::vector<JacobianPoint<Curve>> jacobian_points(size);
    base::Parallelize(
        jacobian_points, [&point, &scalar_fields, affine_points](
                             absl::Span<JacobianPoint<Curve>> chunk,
                             size_t chunk_idx, size_t chunk_size) {
          size_t start = chunk_idx * chunk_size;
          for (size_t i = 0; i < chunk.size(); ++i) {
            chunk[i] = scalar_fields[start + i] * point;
          }
          absl::Span<AffinePoint> sub_affine =
              absl::MakeSpan(*affine_points).subspan(start, chunk.size());
          CHECK(JacobianPoint<Curve>::BatchNormalizeSerial(chunk, &sub_affine));
        });
    return true;
  }

  BaseField x_;
  BaseField y_;
  bool infinity_;
};

}  // namespace math

namespace base {

template <typename Curve>
class Copyable<math::AffinePoint<
    Curve,
    std::enable_if_t<Curve::kType == math::CurveType::kShortWeierstrass>>> {
 public:
  static bool WriteTo(const math::AffinePoint<Curve>& point, Buffer* buffer) {
    return buffer->WriteMany(point.x(), point.y(), point.infinity());
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       math::AffinePoint<Curve>* point) {
    using BaseField = typename math::AffinePoint<Curve>::BaseField;
    BaseField x, y;
    bool infinity;
    if (!buffer.ReadMany(&x, &y, &infinity)) return false;

    *point = math::AffinePoint<Curve>(std::move(x), std::move(y), infinity);
    return true;
  }

  static size_t EstimateSize(const math::AffinePoint<Curve>& point) {
    return base::EstimateSize(point.x(), point.y(), point.infinity());
  }
};

template <typename Curve>
class RapidJsonValueConverter<math::AffinePoint<
    Curve,
    std::enable_if_t<Curve::kType == math::CurveType::kShortWeierstrass>>> {
 public:
  using Field = typename math::AffinePoint<Curve>::BaseField;

  template <typename Allocator>
  static rapidjson::Value From(const math::AffinePoint<Curve>& value,
                               Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    AddJsonElement(object, "x", value.x(), allocator);
    AddJsonElement(object, "y", value.y(), allocator);
    return object;
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 math::AffinePoint<Curve>* value, std::string* error) {
    Field x;
    Field y;
    if (!ParseJsonElement(json_value, "x", &x, error)) return false;
    if (!ParseJsonElement(json_value, "y", &y, error)) return false;
    *value = math::AffinePoint<Curve>(std::move(x), std::move(y));
    return true;
  }
};

}  // namespace base
}  // namespace tachyon

#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point_impl.h"

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_H_
