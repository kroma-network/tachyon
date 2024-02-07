#ifndef TACHYON_CC_MATH_FINITE_FIELDS_PRIME_FIELD_CONVERSIONS_H_
#define TACHYON_CC_MATH_FINITE_FIELDS_PRIME_FIELD_CONVERSIONS_H_

#include <stddef.h>

#include "tachyon/cc/math/finite_fields/prime_field_traits_forward.h"
#include "tachyon/math/base/big_int.h"

namespace tachyon::cc::math {

template <typename CPrimeField, typename PrimeField = typename PrimeFieldTraits<
                                    CPrimeField>::PrimeField>
const PrimeField& native_cast(const CPrimeField& f) {
  static_assert(sizeof(PrimeField) == sizeof(CPrimeField));
  return reinterpret_cast<const PrimeField&>(f);
}

template <typename CPrimeField, typename PrimeField = typename PrimeFieldTraits<
                                    CPrimeField>::PrimeField>
PrimeField& native_cast(CPrimeField& f) {
  static_assert(sizeof(PrimeField) == sizeof(CPrimeField));
  return reinterpret_cast<PrimeField&>(f);
}

template <
    typename PrimeField,
    typename CPrimeField = typename PrimeFieldTraits<PrimeField>::CPrimeField,
    size_t N = PrimeField::N>
const CPrimeField& c_cast(const PrimeField& f) {
  static_assert(sizeof(CPrimeField) == sizeof(PrimeField));
  return reinterpret_cast<const CPrimeField&>(f);
}

template <
    typename PrimeField,
    typename CPrimeField = typename PrimeFieldTraits<PrimeField>::CPrimeField,
    size_t N = PrimeField::N>
CPrimeField& c_cast(PrimeField& f) {
  static_assert(sizeof(CPrimeField) == sizeof(PrimeField));
  return reinterpret_cast<CPrimeField&>(f);
}

}  // namespace tachyon::cc::math

#endif  // TACHYON_CC_MATH_FINITE_FIELDS_PRIME_FIELD_CONVERSIONS_H_
