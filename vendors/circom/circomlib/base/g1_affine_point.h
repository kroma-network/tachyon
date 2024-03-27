#ifndef VENDORS_CIRCOM_CIRCOMLIB_BASE_G1_AFFINE_POINT_H_
#define VENDORS_CIRCOM_CIRCOMLIB_BASE_G1_AFFINE_POINT_H_

#include <stddef.h>

#include <string>
#include <utility>

#include "absl/strings/substitute.h"

#include "circomlib/base/prime_field.h"
#include "tachyon/math/elliptic_curves/affine_point.h"

namespace tachyon::circom {

struct G1AffinePoint {
  PrimeField x;
  PrimeField y;

  bool operator==(const G1AffinePoint& other) const {
    return x == other.x && y == other.y;
  }
  bool operator!=(const G1AffinePoint& other) const {
    return !operator==(other);
  }

  template <bool IsMontgomery, typename Curve>
  math::AffinePoint<Curve> ToNative() const {
    using BaseField = typename math::AffinePoint<Curve>::BaseField;
    BaseField native_x = x.ToNative<IsMontgomery, BaseField>();
    BaseField native_y = y.ToNative<IsMontgomery, BaseField>();
    bool infinity = native_x.IsZero() && native_y.IsZero();
    return {std::move(native_x), std::move(native_y), infinity};
  }

  template <bool IsMontgomery, typename Curve>
  static G1AffinePoint FromNative(const math::AffinePoint<Curve>& point) {
    return {PrimeField::FromNative<IsMontgomery>(point.x()),
            PrimeField::FromNative<IsMontgomery>(point.y())};
  }

  bool Read(const base::ReadOnlyBuffer& buffer, uint32_t field_size) {
    x.bytes.resize(field_size);
    y.bytes.resize(field_size);
    return buffer.Read(x.bytes.data(), field_size) &&
           buffer.Read(y.bytes.data(), field_size);
  }

  template <typename F>
  void Normalize() {
    x.Normalize<F>();
    y.Normalize<F>();
  }

  std::string ToString() const {
    return absl::Substitute("($0, $1)", x.ToString(), y.ToString());
  }
};

}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_CIRCOMLIB_BASE_G1_AFFINE_POINT_H_
