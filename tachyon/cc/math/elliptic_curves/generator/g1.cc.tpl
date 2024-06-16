// clang-format off
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/fq_type_traits.h"
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/g1_point_traits.h"
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/g1_point_type_traits.h"

namespace tachyon::cc::math::%{type} {

std::string G1AffinePoint::ToString() const {
  return c::base::native_cast(ToCPoint()).ToString();
}

std::ostream& operator<<(std::ostream& os, const G1AffinePoint& value) {
  return os << value.ToString();
}

std::string G1ProjectivePoint::ToString() const {
  return c::base::native_cast(ToCPoint()).ToString();
}

std::ostream& operator<<(std::ostream& os, const G1ProjectivePoint& value) {
  return os << value.ToString();
}

std::string G1JacobianPoint::ToString() const {
  return c::base::native_cast(ToCPoint()).ToString();
}

std::ostream& operator<<(std::ostream& os, const G1JacobianPoint& value) {
  return os << value.ToString();
}

std::string G1PointXYZZ::ToString() const {
  return c::base::native_cast(ToCPoint()).ToString();
}

std::ostream& operator<<(std::ostream& os, const G1PointXYZZ& value) {
  return os << value.ToString();
}

std::string G1Point2::ToString() const {
  return c::base::native_cast(ToCPoint()).ToString();
}

std::ostream& operator<<(std::ostream& os, const G1Point2& value) {
  return os << value.ToString();
}

std::string G1Point3::ToString() const {
  return c::base::native_cast(ToCPoint()).ToString();
}

std::ostream& operator<<(std::ostream& os, const G1Point3& value) {
  return os << value.ToString();
}

std::string G1Point4::ToString() const {
  return c::base::native_cast(ToCPoint()).ToString();
}

std::ostream& operator<<(std::ostream& os, const G1Point4& value) {
  return os << value.ToString();
}

} // namespace tachyon::cc::math::%{type}
// clang-format on
