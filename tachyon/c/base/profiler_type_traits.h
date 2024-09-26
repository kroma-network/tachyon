#ifndef TACHYON_C_BASE_PROFILER_TYPE_TRAITS_H_
#define TACHYON_C_BASE_PROFILER_TYPE_TRAITS_H_

#include "tachyon/base/profiler.h"
#include "tachyon/c/base/profiler.h"
#include "tachyon/c/base/type_traits_forward.h"

namespace tachyon::c::base {

template <>
struct TypeTraits<tachyon::base::Profiler> {
  using CType = tachyon_profiler;
};

template <>
struct TypeTraits<tachyon_profiler> {
  using NativeType = tachyon::base::Profiler;
};

}  // namespace tachyon::c::base

#endif  // TACHYON_C_BASE_PROFILER_TYPE_TRAITS_H_
