#ifndef TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_BASE_H_
#define TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_BASE_H_

#include "tachyon/math/base/field.h"

namespace tachyon::math {

template <typename F>
class PrimeFieldBase : public Field<F> {};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_BASE_H_
