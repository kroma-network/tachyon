// clang-format off
#include "tachyon/c/base/type_traits_forward.h"
#include "%{c_hdr}"
#include "%{native_hdr}"

namespace tachyon::c::base {

template <>
struct TypeTraits<%{native_type}> {
  using CType = tachyon_%{class_name};
};

template <>
struct TypeTraits<tachyon_%{class_name}> {
  using NativeType = %{native_type};
};

}  // namespace tachyon::c::base
// clang-format on
