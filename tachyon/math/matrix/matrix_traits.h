#ifndef TACHYON_MATH_MATRIX_MATRIX_TRAITS_H_
#define TACHYON_MATH_MATRIX_MATRIX_TRAITS_H_

#include <stddef.h>

#include <limits>

namespace tachyon {
namespace math {

constexpr size_t kDynamic = std::numeric_limits<size_t>::max();

template <typename T>
struct MatrixTraits;

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_MATRIX_MATRIX_TRAITS_H_