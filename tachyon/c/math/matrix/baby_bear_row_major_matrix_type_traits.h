#ifndef TACHYON_C_MATH_MATRIX_BABY_BEAR_ROW_MAJOR_MATRIX_TYPE_TRAITS_H_
#define TACHYON_C_MATH_MATRIX_BABY_BEAR_ROW_MAJOR_MATRIX_TYPE_TRAITS_H_

#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/math/finite_fields/baby_bear/baby_bear.h"
#include "tachyon/math/matrix/matrix_types.h"

namespace tachyon::c::base {

template <>
struct TypeTraits<tachyon::math::RowMajorMatrix<tachyon::math::BabyBear>> {
  using CType = tachyon_baby_bear_row_major_matrix;
};

template <>
struct TypeTraits<tachyon_baby_bear_row_major_matrix> {
  using NativeType = tachyon::math::RowMajorMatrix<tachyon::math::BabyBear>;
};

}  // namespace tachyon::c::base

#endif  // TACHYON_C_MATH_MATRIX_BABY_BEAR_ROW_MAJOR_MATRIX_TYPE_TRAITS_H_
