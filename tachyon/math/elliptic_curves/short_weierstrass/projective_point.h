#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_PROJECTIVE_POINT_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_PROJECTIVE_POINT_H_

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/math/base/groups.h"
#include "tachyon/math/geometry/affine_point.h"
#include "tachyon/math/geometry/curve_type.h"
#include "tachyon/math/geometry/jacobian_point.h"
#include "tachyon/math/geometry/point3.h"
#include "tachyon/math/geometry/point_xyzz.h"
#include "tachyon/math/geometry/projective_point.h"

namespace tachyon {
namespace math {

template <typename _Curve>
class ProjectivePoint<
    _Curve, std::enable_if_t<_Curve::kType == CurveType::kShortWeierstrass>>
    final : public AdditiveGroup<ProjectivePoint<_Curve>> {
 public:
  constexpr static bool kNegationIsCheap = true;

  using Curve = _Curve;
  using BaseField = typename Curve::BaseField;
  using ScalarField = typename Curve::ScalarField;

  constexpr ProjectivePoint()
      : ProjectivePoint(BaseField::One(), BaseField::One(), BaseField::Zero()) {
  }
  explicit constexpr ProjectivePoint(const Point3<BaseField>& point)
      : ProjectivePoint(point.x, point.y, point.z) {}
  explicit constexpr ProjectivePoint(Point3<BaseField>&& point)
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

  constexpr static std::optional<ProjectivePoint> CreateFromX(
      const BaseField& x, bool pick_odd) {
    ProjectivePoint point{};
    if (!Curve::GetPointFromX(x, pick_odd, &point)) return std::nullopt;
    return point;
  }

  constexpr static ProjectivePoint Zero() { return ProjectivePoint(); }

  constexpr static ProjectivePoint One() { return Generator(); }

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

  constexpr static ProjectivePoint Random() {
    return ScalarField::Random() * Generator();
  }

  constexpr static ProjectivePoint Endomorphism(const ProjectivePoint& point) {
    return ProjectivePoint(point.x_ * Curve::Config::kEndomorphismCoefficient,
                           point.y_, point.z_);
  }

  template <typename ProjectiveContainer, typename AffineContainer>
  [[nodiscard]] CONSTEXPR_IF_NOT_OPENMP static bool BatchNormalize(
      const ProjectiveContainer& projective_points,
      AffineContainer* affine_points) {
    size_t size = std::size(projective_points);
    if (size != std::size(*affine_points)) {
      LOG(ERROR)
          << "Size of |projective_points| and |affine_points| do not match";
      return false;
    }
    std::vector<BaseField> z_inverses =
        base::Map(projective_points,
                  [](const ProjectivePoint& point) { return point.z_; });
#if defined(TACHYON_HAS_OPENMP)
    size_t thread_nums = static_cast<size_t>(omp_get_max_threads());
    if (size >=
        size_t{1} << (thread_nums /
                      ScalarField::kParallelBatchInverseDivisorThreshold)) {
      size_t chunk_size = base::GetNumElementsPerThread(projective_points);
      size_t num_chunks = (size + chunk_size - 1) / chunk_size;
      OPENMP_PARALLEL_FOR(size_t i = 0; i < num_chunks; ++i) {
        size_t len = i == num_chunks - 1 ? size - i * chunk_size : chunk_size;
        absl::Span<AffinePoint<Curve>> affine_points_chunk(
            std::data(*affine_points) + i * chunk_size, len);
        absl::Span<const ProjectivePoint> projective_points_chunk(
            std::data(projective_points) + i * chunk_size, len);
        absl::Span<BaseField> z_inverses_chunk(&z_inverses[i * chunk_size],
                                               len);

        CHECK(BaseField::BatchInverseInPlaceSerial(z_inverses_chunk));
        for (size_t i = 0; i < z_inverses_chunk.size(); ++i) {
          const BaseField& z_inv = z_inverses_chunk[i];
          if (z_inv.IsZero()) {
            affine_points_chunk[i] = AffinePoint<Curve>::Zero();
          } else if (z_inv.IsOne()) {
            affine_points_chunk[i] = {projective_points_chunk[i].x_,
                                      projective_points_chunk[i].y_};
          } else {
            affine_points_chunk[i] = {projective_points_chunk[i].x_ * z_inv,
                                      projective_points_chunk[i].y_ * z_inv};
          }
        }
      }
      return true;
    }
#endif
    return BatchNormalizeSerial(projective_points, affine_points);
  }

  template <typename ProjectiveContainer, typename AffineContainer>
  [[nodiscard]] constexpr static bool BatchNormalizeSerial(
      const ProjectiveContainer& projective_points,
      AffineContainer* affine_points) {
    size_t size = std::size(projective_points);
    if (size != std::size(*affine_points)) {
      LOG(ERROR)
          << "Size of |projective_points| and |affine_points| do not match";
      return false;
    }
    std::vector<BaseField> z_inverses =
        base::Map(projective_points,
                  [](const ProjectivePoint& point) { return point.z_; });
    if (!BaseField::BatchInverseInPlaceSerial(z_inverses)) return false;
    for (size_t i = 0; i < size; ++i) {
      const BaseField& z_inv = z_inverses[i];
      if (z_inv.IsZero()) {
        (*affine_points)[i] = AffinePoint<Curve>::Zero();
      } else if (z_inv.IsOne()) {
        (*affine_points)[i] = {projective_points[i].x_,
                               projective_points[i].y_};
      } else {
        (*affine_points)[i] = {projective_points[i].x_ * z_inv,
                               projective_points[i].y_ * z_inv};
      }
    }
    return true;
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
      // NOTE(ashjeong): if |z_| is 0, |IsZero()| will also evaluate to true,
      // and this block will not be executed
      BaseField z_inv = *z_.Inverse();
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

  std::string ToString() const {
    return absl::Substitute("($0, $1, $2)", x_.ToString(), y_.ToString(),
                            z_.ToString());
  }

  std::string ToHexString(bool pad_zero = false) const {
    return absl::Substitute("($0, $1, $2)", x_.ToHexString(pad_zero),
                            y_.ToHexString(pad_zero), z_.ToHexString(pad_zero));
  }

  // AdditiveSemigroup methods
  constexpr ProjectivePoint Add(const ProjectivePoint& other) const;
  constexpr ProjectivePoint& AddInPlace(const ProjectivePoint& other);
  constexpr ProjectivePoint Add(const AffinePoint<Curve>& other) const;
  constexpr ProjectivePoint& AddInPlace(const AffinePoint<Curve>& other);
  constexpr ProjectivePoint DoubleImpl() const;
  constexpr ProjectivePoint& DoubleImplInPlace();

  // AdditiveGroup methods
  constexpr ProjectivePoint Negate() const { return {x_, -y_, z_}; }

  constexpr ProjectivePoint& NegateInPlace() {
    y_.NegateInPlace();
    return *this;
  }

  constexpr ProjectivePoint operator*(const ScalarField& v) const {
    return this->ScalarMul(v);
  }
  constexpr ProjectivePoint& operator*=(const ScalarField& v) {
    return *this = operator*(v);
  }

 private:
  constexpr static void DoAdd(const ProjectivePoint& a,
                              const ProjectivePoint& b, ProjectivePoint& c);
  constexpr static void DoAdd(const ProjectivePoint& a,
                              const AffinePoint<Curve>& b, ProjectivePoint& c);
  constexpr static void DoDoubleImpl(const ProjectivePoint& a,
                                     ProjectivePoint& b);

  BaseField x_;
  BaseField y_;
  BaseField z_;
};

}  // namespace math

namespace base {

template <typename Curve>
class Copyable<math::ProjectivePoint<
    Curve,
    std::enable_if_t<Curve::kType == math::CurveType::kShortWeierstrass>>> {
 public:
  static bool WriteTo(const math::ProjectivePoint<Curve>& point,
                      Buffer* buffer) {
    return buffer->WriteMany(point.x(), point.y(), point.z());
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       math::ProjectivePoint<Curve>* point) {
    using BaseField = typename math::ProjectivePoint<Curve>::BaseField;
    BaseField x, y, z;
    if (!buffer.ReadMany(&x, &y, &z)) return false;

    *point =
        math::ProjectivePoint<Curve>(std::move(x), std::move(y), std::move(z));
    return true;
  }

  static size_t EstimateSize(const math::ProjectivePoint<Curve>& point) {
    return base::EstimateSize(point.x(), point.y(), point.z());
  }
};

template <typename Curve>
class RapidJsonValueConverter<math::ProjectivePoint<
    Curve,
    std::enable_if_t<Curve::kType == math::CurveType::kShortWeierstrass>>> {
 public:
  using Field = typename math::ProjectivePoint<Curve>::BaseField;

  template <typename Allocator>
  static rapidjson::Value From(const math::ProjectivePoint<Curve>& value,
                               Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    AddJsonElement(object, "x", value.x(), allocator);
    AddJsonElement(object, "y", value.y(), allocator);
    AddJsonElement(object, "z", value.z(), allocator);
    return object;
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 math::ProjectivePoint<Curve>* value, std::string* error) {
    Field x;
    Field y;
    Field z;
    if (!ParseJsonElement(json_value, "x", &x, error)) return false;
    if (!ParseJsonElement(json_value, "y", &y, error)) return false;
    if (!ParseJsonElement(json_value, "z", &z, error)) return false;
    *value =
        math::ProjectivePoint<Curve>(std::move(x), std::move(y), std::move(z));
    return true;
  }
};

}  // namespace base
}  // namespace tachyon

#include "tachyon/math/elliptic_curves/short_weierstrass/projective_point_impl.h"

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_PROJECTIVE_POINT_H_
