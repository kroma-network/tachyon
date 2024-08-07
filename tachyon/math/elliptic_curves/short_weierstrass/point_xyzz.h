#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_POINT_XYZZ_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_POINT_XYZZ_H_

#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/json/json.h"
#include "tachyon/base/logging.h"
#include "tachyon/math/base/groups.h"
#include "tachyon/math/geometry/affine_point.h"
#include "tachyon/math/geometry/curve_type.h"
#include "tachyon/math/geometry/jacobian_point.h"
#include "tachyon/math/geometry/point4.h"
#include "tachyon/math/geometry/point_xyzz.h"
#include "tachyon/math/geometry/projective_point.h"

namespace tachyon {
namespace math {

template <typename _Curve>
class PointXYZZ<_Curve,
                std::enable_if_t<_Curve::kType == CurveType::kShortWeierstrass>>
    final : public AdditiveGroup<PointXYZZ<_Curve>> {
 public:
  constexpr static bool kNegationIsCheap = true;

  using Curve = _Curve;
  using BaseField = typename Curve::BaseField;
  using ScalarField = typename Curve::ScalarField;

  constexpr PointXYZZ()
      : PointXYZZ(BaseField::One(), BaseField::One(), BaseField::Zero(),
                  BaseField::Zero()) {}
  explicit constexpr PointXYZZ(const Point4<BaseField>& point)
      : PointXYZZ(point.x, point.y, point.z, point.w) {}
  explicit constexpr PointXYZZ(Point4<BaseField>&& point)
      : PointXYZZ(std::move(point.x), std::move(point.y), std::move(point.z),
                  std::move(point.w)) {}
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
    CHECK(ret.IsOnCurve());
    return ret;
  }

  constexpr static PointXYZZ CreateChecked(BaseField&& x, BaseField&& y,
                                           BaseField&& zz, BaseField&& zzz) {
    PointXYZZ ret = {std::move(x), std::move(y), std::move(zz), std::move(zzz)};
    CHECK(ret.IsOnCurve());
    return ret;
  }

  constexpr static std::optional<PointXYZZ> CreateFromX(const BaseField& x,
                                                        bool pick_odd) {
    PointXYZZ point{};
    if (!Curve::GetPointFromX(x, pick_odd, &point)) return std::nullopt;
    return point;
  }

  constexpr static PointXYZZ Zero() { return PointXYZZ(); }

  constexpr static PointXYZZ One() { return Generator(); }

  constexpr static PointXYZZ Generator() {
    return {Curve::Config::kGenerator.x, Curve::Config::kGenerator.y,
            BaseField::One(), BaseField::One()};
  }

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

  constexpr static PointXYZZ Random() {
    return FromJacobian(JacobianPoint<Curve>::Random());
  }

  constexpr static PointXYZZ Endomorphism(const PointXYZZ& point) {
    return PointXYZZ(point.x_ * Curve::Config::kEndomorphismCoefficient,
                     point.y_, point.zz_, point.zzz_);
  }

  template <typename PointXYZZContainer, typename AffineContainer>
  [[nodiscard]] CONSTEXPR_IF_NOT_OPENMP static bool BatchNormalize(
      const PointXYZZContainer& point_xyzzs, AffineContainer* affine_points) {
    size_t size = std::size(point_xyzzs);
    if (size != std::size(*affine_points)) {
      LOG(ERROR) << "Size of |point_xyzzs| and |affine_points| do not match";
      return false;
    }
#if defined(TACHYON_HAS_OPENMP)
    size_t thread_nums = static_cast<size_t>(omp_get_max_threads());
    if (size >=
        size_t{1} << (thread_nums /
                      ScalarField::kParallelBatchInverseDivisorThreshold)) {
      size_t chunk_size = base::GetNumElementsPerThread(point_xyzzs);
      size_t num_chunks = (size + chunk_size - 1) / chunk_size;
      OMP_PARALLEL_FOR(size_t i = 0; i < num_chunks; ++i) {
        size_t len = i == num_chunks - 1 ? size - i * chunk_size : chunk_size;
        absl::Span<AffinePoint<Curve>> affine_points_chunk(
            std::data(*affine_points) + i * chunk_size, len);
        absl::Span<const PointXYZZ> point_xyzzs_chunk(
            std::data(point_xyzzs) + i * chunk_size, len);
        CHECK(BatchNormalizeSerial(point_xyzzs_chunk, &affine_points_chunk));
      }
      return true;
    }
#endif
    return BatchNormalizeSerial(point_xyzzs, affine_points);
  }

  template <typename PointXYZZContainer, typename AffineContainer>
  [[nodiscard]] constexpr static bool BatchNormalizeSerial(
      const PointXYZZContainer& point_xyzzs, AffineContainer* affine_points) {
    size_t size = std::size(point_xyzzs);
    if (size != std::size(*affine_points)) {
      LOG(ERROR) << "Size of |point_xyzzs| and |affine_points| do not match";
      return false;
    }
    std::vector<BaseField> zzz_inverses = base::Map(
        point_xyzzs, [](const PointXYZZ& point) { return point.zzz_; });
    if (!BaseField::BatchInverseInPlaceSerial(zzz_inverses)) return false;
    for (size_t i = 0; i < size; ++i) {
      const PointXYZZ& point_xyzz = point_xyzzs[i];
      if (point_xyzz.zz_.IsZero()) {
        (*affine_points)[i] = AffinePoint<Curve>::Zero();
      } else if (point_xyzz.zz_.IsOne()) {
        (*affine_points)[i] = {point_xyzz.x_, point_xyzz.y_};
      } else {
        const BaseField& z_inv_cubic = zzz_inverses[i];
        BaseField z_inv_square = z_inv_cubic * point_xyzz.zz_;
        z_inv_square.SquareInPlace();
        (*affine_points)[i] = {point_xyzz.x_ * z_inv_square,
                               point_xyzz.y_ * z_inv_cubic};
      }
    }
    return true;
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

  constexpr bool IsOnCurve() { return Curve::IsOnCurve(*this); }

  // The xyzz point X, Y, ZZ, ZZZ is represented in the affine
  // coordinates as X/ZZ, Y/ZZZ.
  constexpr AffinePoint<Curve> ToAffine() const {
    if (IsZero()) {
      return AffinePoint<Curve>::Zero();
    } else if (zz_.IsOne()) {
      return {x_, y_};
    } else {
      // NOTE(ashjeong): if |zzz_| is 0, |IsZero()| will also evaluate to true,
      // and this block will not be executed
      BaseField z_inv_cubic = *zzz_.Inverse();
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

  std::string ToString() const {
    return absl::Substitute("($0, $1, $2, $3)", x_.ToString(), y_.ToString(),
                            zz_.ToString(), zzz_.ToString());
  }

  std::string ToHexString(bool pad_zero = false) const {
    return absl::Substitute("($0, $1, $2, $3)", x_.ToHexString(pad_zero),
                            y_.ToHexString(pad_zero), zz_.ToHexString(pad_zero),
                            zzz_.ToHexString(pad_zero));
  }

  // AdditiveSemigroup methods
  constexpr PointXYZZ Add(const PointXYZZ& other) const;
  constexpr PointXYZZ& AddInPlace(const PointXYZZ& other);
  constexpr PointXYZZ Add(const AffinePoint<Curve>& other) const;
  constexpr PointXYZZ& AddInPlace(const AffinePoint<Curve>& other);
  constexpr PointXYZZ DoubleImpl() const;
  constexpr PointXYZZ& DoubleImplInPlace();

  // AdditiveGroup methods
  constexpr PointXYZZ Negate() const { return {x_, -y_, zz_, zzz_}; }

  constexpr PointXYZZ& NegateInPlace() {
    y_.NegateInPlace();
    return *this;
  }

  constexpr PointXYZZ operator*(const ScalarField& v) const {
    return this->ScalarMul(v);
  }
  constexpr PointXYZZ& operator*=(const ScalarField& v) {
    return *this = operator*(v);
  }

 private:
  constexpr static void DoAdd(const PointXYZZ& a, const PointXYZZ& b,
                              PointXYZZ& c);
  CONSTEXPR_IF_NOT_OPENMP static void DoAdd(const PointXYZZ& a,
                                            const AffinePoint<Curve>& b,
                                            PointXYZZ& c);
  constexpr static void DoDoubleImpl(const PointXYZZ& a, PointXYZZ& b);

  BaseField x_;
  BaseField y_;
  BaseField zz_;
  BaseField zzz_;
};

}  // namespace math

namespace base {

template <typename Curve>
class Copyable<math::PointXYZZ<
    Curve,
    std::enable_if_t<Curve::kType == math::CurveType::kShortWeierstrass>>> {
 public:
  static bool WriteTo(const math::PointXYZZ<Curve>& point, Buffer* buffer) {
    return buffer->WriteMany(point.x(), point.y(), point.zz(), point.zzz());
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       math::PointXYZZ<Curve>* point) {
    using BaseField = typename math::PointXYZZ<Curve>::BaseField;
    BaseField x, y, zz, zzz;
    if (!buffer.ReadMany(&x, &y, &zz, &zzz)) return false;

    *point = math::PointXYZZ<Curve>(std::move(x), std::move(y), std::move(zz),
                                    std::move(zzz));
    return true;
  }

  static size_t EstimateSize(const math::PointXYZZ<Curve>& point) {
    return base::EstimateSize(point.x(), point.y(), point.zz(), point.zzz());
  }
};

template <typename Curve>
class RapidJsonValueConverter<math::PointXYZZ<
    Curve,
    std::enable_if_t<Curve::kType == math::CurveType::kShortWeierstrass>>> {
 public:
  using Field = typename math::PointXYZZ<Curve>::BaseField;

  template <typename Allocator>
  static rapidjson::Value From(const math::PointXYZZ<Curve>& value,
                               Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    AddJsonElement(object, "x", value.x(), allocator);
    AddJsonElement(object, "y", value.y(), allocator);
    AddJsonElement(object, "zz", value.zz(), allocator);
    AddJsonElement(object, "zzz", value.zzz(), allocator);
    return object;
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 math::PointXYZZ<Curve>* value, std::string* error) {
    Field x;
    Field y;
    Field zz;
    Field zzz;
    if (!ParseJsonElement(json_value, "x", &x, error)) return false;
    if (!ParseJsonElement(json_value, "y", &y, error)) return false;
    if (!ParseJsonElement(json_value, "zz", &zz, error)) return false;
    if (!ParseJsonElement(json_value, "zzz", &zzz, error)) return false;
    *value = math::PointXYZZ<Curve>(std::move(x), std::move(y), std::move(zz),
                                    std::move(zzz));
    return true;
  }
};

}  // namespace base
}  // namespace tachyon

#include "tachyon/math/elliptic_curves/short_weierstrass/point_xyzz_impl.h"

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_POINT_XYZZ_H_
