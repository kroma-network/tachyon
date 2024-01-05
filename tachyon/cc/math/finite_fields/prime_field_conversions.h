#ifndef TACHYON_CC_MATH_FINITE_FIELDS_PRIME_FIELD_CONVERSIONS_H_
#define TACHYON_CC_MATH_FINITE_FIELDS_PRIME_FIELD_CONVERSIONS_H_

#include <stddef.h>

#include "tachyon/cc/math/finite_fields/prime_field_traits_forward.h"
#include "tachyon/math/base/big_int.h"

namespace tachyon::cc::math {

template <
    typename CPrimeField,
    typename PrimeField = typename PrimeFieldTraits<CPrimeField>::PrimeField,
    size_t N = PrimeField::N>
tachyon::math::BigInt<N> ToBigInt(const CPrimeField& f) {
  return tachyon::math::BigInt<N>(f.limbs);
}

template <typename CPrimeField, typename PrimeField = typename PrimeFieldTraits<
                                    CPrimeField>::PrimeField>
PrimeField ToPrimeField(const CPrimeField& f) {
  return PrimeField::FromMontgomery(ToBigInt(f));
}

template <
    typename PrimeField,
    typename CPrimeField = typename PrimeFieldTraits<PrimeField>::CPrimeField,
    size_t N = PrimeField::N>
CPrimeField ToCPrimeField(const PrimeField& f) {
  CPrimeField ret;
  memcpy(ret.limbs, f.value().limbs, sizeof(uint64_t) * N);
  return ret;
}

}  // namespace tachyon::cc::math

#endif  // TACHYON_CC_MATH_FINITE_FIELDS_PRIME_FIELD_CONVERSIONS_H_
