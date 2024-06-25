#ifndef TACHYON_C_MATH_FINITE_FIELDS_BABY_BEAR_BABY_BEAR_TYPE_TRAITS_H_
#define TACHYON_C_MATH_FINITE_FIELDS_BABY_BEAR_BABY_BEAR_TYPE_TRAITS_H_

#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/math/finite_fields/baby_bear/baby_bear.h"
#include "tachyon/math/finite_fields/baby_bear/baby_bear.h"

namespace tachyon::c::base {

template <>
struct TypeTraits<tachyon::math::BabyBear> {
  using CType = tachyon_baby_bear;
};

template <>
struct TypeTraits<tachyon_baby_bear> {
  using NativeType = tachyon::math::BabyBear;
};

}  // namespace tachyon::c::base

#endif  // TACHYON_C_MATH_FINITE_FIELDS_BABY_BEAR_BABY_BEAR_TYPE_TRAITS_H_
