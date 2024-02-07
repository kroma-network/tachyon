#ifndef TACHYON_C_MATH_POLYNOMIALS_CONSTANTS_H_
#define TACHYON_C_MATH_POLYNOMIALS_CONSTANTS_H_

#include <stddef.h>
#include <stdint.h>

namespace tachyon::c::math {

// NOTE(chokobole): We set |kMaxDegree| to |SIZE_MAX| - 1 on purpose to avoid
// creating variant apis corresponding to the set of each degree.
constexpr size_t kMaxDegree = SIZE_MAX - 1;

}  // namespace tachyon::c::math

#endif  // TACHYON_C_MATH_POLYNOMIALS_CONSTANTS_H_
