// clang-format off
#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/%{suffix}.h"
#include "tachyon/math/elliptic_curves/%{header_dir_name}/%{suffix}.h"

namespace tachyon::c::base {

template <>
struct TypeTraits<tachyon_%{type}_%{suffix}> {
  using NativeType = tachyon::math::%{type}::%{Suffix};
};

template <>
struct TypeTraits<tachyon::math::%{type}::%{Suffix}> {
  using CType = tachyon_%{type}_%{suffix};
};

}  // namespace tachyon::cc::math
// clang-format on
