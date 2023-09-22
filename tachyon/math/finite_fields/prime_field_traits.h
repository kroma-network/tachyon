#ifndef TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_TRAITS_H_
#define TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_TRAITS_H_

#include "tachyon/math/finite_fields/prime_field_forward.h"

namespace tachyon::math {

template <typename T>
struct PrimeFieldTraits {
  static constexpr bool kIsPrimeField = false;
};

template <typename _Config>
struct PrimeFieldTraits<PrimeField<_Config>> {
  static constexpr bool kIsPrimeField = true;

  using Config = _Config;
};

template <typename _Config>
struct PrimeFieldTraits<PrimeFieldGmp<_Config>> {
  static constexpr bool kIsPrimeField = true;

  using Config = _Config;
};

template <typename _Config>
struct PrimeFieldTraits<PrimeFieldGpu<_Config>> {
  static constexpr bool kIsPrimeField = true;

  using Config = _Config;
};

template <typename _Config>
struct PrimeFieldTraits<PrimeFieldGpuDebug<_Config>> {
  static constexpr bool kIsPrimeField = true;

  using Config = _Config;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_TRAITS_H_
