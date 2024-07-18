// clang-format off
#include <string.h>

#include <ostream>

#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/%{suffix}.h"
#include "tachyon/cc/export.h"

namespace tachyon::cc::math::%{type} {

class TACHYON_CC_EXPORT %{cc_field} {
 public:
  %{cc_field}() = default;
  explicit %{cc_field}(%{c_field} value) {
    memcpy(value_.limbs, value.limbs, sizeof(uint64_t) * %{limb_nums});
  }

  const %{c_field}& value() const { return value_; }
  %{c_field}& value() { return value_; }

  const %{c_field}* value_ptr() const { return &value_; }
  %{c_field}* value_ptr() { return &value_; }

  static %{cc_field} Zero() {
    return %{cc_field}(%{c_field}_zero());
  }

  static %{cc_field} One() {
    return %{cc_field}(%{c_field}_one());
  }

  static %{cc_field} MinusOne() {
    return %{cc_field}(%{c_field}_minus_one());
  }

  static %{cc_field} Random() {
    return %{cc_field}(%{c_field}_random());
  }

  %{cc_field} operator+(const %{cc_field}& other) const {
    return %{cc_field}(%{c_field}_add(&value_, &other.value_));
  }

  %{cc_field}& operator+=(const %{cc_field}& other) {
    value_ = %{c_field}_add(&value_, &other.value_);
    return *this;
  }

  %{cc_field} operator-(const %{cc_field}& other) const {
    return %{cc_field}(%{c_field}_sub(&value_, &other.value_));
  }

  %{cc_field}& operator-=(const %{cc_field}& other) {
    value_ = %{c_field}_sub(&value_, &other.value_);
    return *this;
  }

  %{cc_field} operator*(const %{cc_field}& other) const {
    return %{cc_field}(%{c_field}_mul(&value_, &other.value_));
  }

  %{cc_field}& operator*=(const %{cc_field}& other) {
    value_ = %{c_field}_mul(&value_, &other.value_);
    return *this;
  }

  %{cc_field} operator/(const %{cc_field}& other) const {
    return %{cc_field}(%{c_field}_div(&value_, &other.value_));
  }

  %{cc_field}& operator/=(const %{cc_field}& other) {
    value_ = %{c_field}_div(&value_, &other.value_);
    return *this;
  }

  %{cc_field} operator-() const {
    return %{cc_field}(%{c_field}_neg(&value_));
  }

  %{cc_field} Double() const {
    return %{cc_field}(%{c_field}_dbl(&value_));
  }

  %{cc_field} Square() const {
    return %{cc_field}(%{c_field}_sqr(&value_));
  }

  %{cc_field} Inverse() const {
    return %{cc_field}(%{c_field}_inv(&value_));
  }

  bool operator==(const %{cc_field}& other) const {
    return %{c_field}_eq(&value_, &other.value_);
  }

  bool operator!=(const %{cc_field}& other) const {
    return %{c_field}_ne(&value_, &other.value_);
  }

  bool operator>(const %{cc_field}& other) const {
    return %{c_field}_gt(&value_, &other.value_);
  }

  bool operator>=(const %{cc_field}& other) const {
    return %{c_field}_ge(&value_, &other.value_);
  }

  bool operator<(const %{cc_field}& other) const {
    return %{c_field}_lt(&value_, &other.value_);
  }

  bool operator<=(const %{cc_field}& other) const {
    return %{c_field}_le(&value_, &other.value_);
  }

  std::string ToString() const;

 private:
  %{c_field} value_;
};

TACHYON_CC_EXPORT std::ostream& operator<<(std::ostream& os, const %{cc_field}& value);

} // namespace tachyon::cc::math::%{type}
// clang-format on
