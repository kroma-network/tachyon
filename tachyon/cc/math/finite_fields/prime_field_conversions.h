#ifndef TACHYON_CC_MATH_FINITE_FIELDS_PRIME_FIELD_CONVERSIONS_H_
#define TACHYON_CC_MATH_FINITE_FIELDS_PRIME_FIELD_CONVERSIONS_H_

#include <stddef.h>

#include "tachyon/cc/math/finite_fields/prime_field_traits.h"
#include "tachyon/math/base/big_int.h"

namespace tachyon::cc::math {

template <typename CPrimeFieldTy,
          typename PrimeFieldTy =
              typename PrimeFieldTraits<CPrimeFieldTy>::PrimeFieldTy,
          size_t N = PrimeFieldTy::N>
tachyon::math::BigInt<N> ToBigInt(const CPrimeFieldTy& f) {
  return tachyon::math::BigInt<N>(f.limbs);
}

template <typename CPrimeFieldTy,
          typename PrimeFieldTy =
              typename PrimeFieldTraits<CPrimeFieldTy>::PrimeFieldTy>
PrimeFieldTy ToPrimeField(const CPrimeFieldTy& f) {
  return PrimeFieldTy::FromMontgomery(ToBigInt(f));
}

template <typename PrimeFieldTy,
          typename CPrimeFieldTy =
              typename PrimeFieldTraits<PrimeFieldTy>::CPrimeFieldTy,
          size_t N = PrimeFieldTy::N>
CPrimeFieldTy ToCPrimeField(const PrimeFieldTy& f) {
  CPrimeFieldTy ret;
  memcpy(ret.limbs, f.value().limbs, sizeof(uint64_t) * N);
  return ret;
}

}  // namespace tachyon::cc::math

#endif  // TACHYON_CC_MATH_FINITE_FIELDS_PRIME_FIELD_CONVERSIONS_H_
