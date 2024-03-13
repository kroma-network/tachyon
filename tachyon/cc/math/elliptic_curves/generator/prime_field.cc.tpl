// clang-format off
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/%{suffix}_prime_field_traits.h"
#include "tachyon/cc/math/finite_fields/prime_field_conversions.h"

namespace tachyon::cc::math::%{type} {

std::string %{cc_field}::ToString() const {
  return native_cast(value_).ToString();
}

std::ostream& operator<<(std::ostream& os, const %{cc_field}& value) {
  return os << value.ToString();
}

} // namespace tachyon::cc::math::%{type}
// clang-format on
