#ifndef VENDORS_CIRCOM_CIRCOMLIB_BASE_FR_ELEMENT_CONVERSION_H_
#define VENDORS_CIRCOM_CIRCOMLIB_BASE_FR_ELEMENT_CONVERSION_H_

#include <string.h>

#include <utility>

#include "fr.hpp"  // NOLINT(build/include_subdir)

namespace tachyon::circom {

template <typename F>
FrElement ConvertToFrElement(const F& value) {
  FrElement fr;
  fr.type = Fr_LONGMONTGOMERY;
  memcpy(fr.longVal, value.value().limbs, sizeof(uint64_t) * F::kLimbNums);
  return fr;
}

template <typename F>
F ConvertFromFrElement(FrElement& value) {
  using BigInt = typename F::BigIntTy;
  FrElement tmp;
  Fr_toLongNormal(&tmp, &value);
  BigInt bigint;
  memcpy(bigint.limbs, tmp.longVal, sizeof(uint64_t) * F::kLimbNums);
  return F(std::move(bigint));
}

}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_CIRCOMLIB_BASE_FR_ELEMENT_CONVERSION_H_
