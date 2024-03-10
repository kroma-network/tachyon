#ifndef VENDORS_CIRCOM_CIRCOMLIB_BASE_G2_AFFINE_POINT_H_
#define VENDORS_CIRCOM_CIRCOMLIB_BASE_G2_AFFINE_POINT_H_

#include <stddef.h>

#include <array>
#include <string>
#include <utility>

#include "absl/strings/substitute.h"

#include "circomlib/base/prime_field.h"
#include "tachyon/math/elliptic_curves/affine_point.h"

namespace tachyon::circom {

struct G2AffinePoint {
  std::array<PrimeField, 2> x;
  std::array<PrimeField, 2> y;

  bool operator==(const G2AffinePoint& other) const {
    return x == other.x && y == other.y;
  }
  bool operator!=(const G2AffinePoint& other) const {
    return !operator==(other);
  }

  template <typename Curve>
  math::AffinePoint<Curve> ToNative() const {
    using BaseField = typename math::AffinePoint<Curve>::BaseField;
    using BasePrimeField = typename BaseField::BasePrimeField;
    BaseField native_x(x[0].ToNative<BasePrimeField>(),
                       x[1].ToNative<BasePrimeField>());
    BaseField native_y(y[0].ToNative<BasePrimeField>(),
                       y[1].ToNative<BasePrimeField>());
    bool infinity = native_x.IsZero() && native_y.IsZero();
    return {std::move(native_x), std::move(native_y), infinity};
  }

  template <typename Curve>
  static G2AffinePoint FromNative(const math::AffinePoint<Curve>& point) {
    return {{PrimeField::FromNative(point.x().c0()),
             PrimeField::FromNative(point.x().c1())},
            {PrimeField::FromNative(point.y().c0()),
             PrimeField::FromNative(point.y().c1())}};
  }

  bool Read(const base::ReadOnlyBuffer& buffer, uint32_t field_size) {
    x[0].bytes.resize(field_size);
    x[1].bytes.resize(field_size);
    y[0].bytes.resize(field_size);
    y[1].bytes.resize(field_size);
    return buffer.Read(x[0].bytes.data(), field_size) &&
           buffer.Read(x[1].bytes.data(), field_size) &&
           buffer.Read(y[0].bytes.data(), field_size) &&
           buffer.Read(y[1].bytes.data(), field_size);
  }

  std::string ToString() const {
    return absl::Substitute("(($0, $1), ($2, $3))", x[0].ToString(),
                            x[1].ToString(), y[0].ToString(), y[1].ToString());
  }
};

}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_CIRCOMLIB_BASE_G2_AFFINE_POINT_H_
