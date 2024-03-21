// clang-format off
#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/fq%{degree}.h"
#include "tachyon/math/elliptic_curves/%{header_dir_name}/fq%{degree}.h"

namespace tachyon::c::base {

template <>
struct TypeTraits<tachyon::math::%{type}::Fq%{degree}> {
  using CType = tachyon_%{type}_fq%{degree};
};

template <>
struct TypeTraits<tachyon_%{type}_fq%{degree}> {
  using NativeType = tachyon::math::%{type}::Fq%{degree};
};

}  // namespace tachyon::cc::math
// clang-format on
