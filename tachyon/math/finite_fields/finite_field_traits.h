#ifndef TACHYON_MATH_FINITE_FIELDS_FINITE_FIELD_TRAITS_H_
#define TACHYON_MATH_FINITE_FIELDS_FINITE_FIELD_TRAITS_H_

#include "tachyon/math/finite_fields/finite_field_forwards.h"

namespace tachyon::math {

template <typename T>
struct FiniteFieldTraits {
  static constexpr bool kIsPrimeField = false;
  static constexpr bool kIsPackedPrimeField = false;
  static constexpr bool kIsExtensionField = false;
};

template <typename _Config>
struct FiniteFieldTraits<BinaryField<_Config>> {
  static constexpr bool kIsPrimeField = false;
  static constexpr bool kIsPackedPrimeField = false;
  static constexpr bool kIsExtensionField = true;

  using Config = _Config;
};

template <typename _Config>
struct FiniteFieldTraits<PrimeField<_Config>> {
  static constexpr bool kIsPrimeField = true;
  static constexpr bool kIsPackedPrimeField = false;
  static constexpr bool kIsExtensionField = false;

  using Config = _Config;
};

template <typename _Config>
struct FiniteFieldTraits<PrimeFieldGpu<_Config>> {
  static constexpr bool kIsPrimeField = true;
  static constexpr bool kIsPackedPrimeField = false;
  static constexpr bool kIsExtensionField = false;

  using Config = _Config;
};

template <typename _Config>
struct FiniteFieldTraits<PrimeFieldGpuDebug<_Config>> {
  static constexpr bool kIsPrimeField = true;
  static constexpr bool kIsPackedPrimeField = false;
  static constexpr bool kIsExtensionField = false;

  using Config = _Config;
};

template <typename _Config>
struct FiniteFieldTraits<Fp2<_Config>> {
  static constexpr bool kIsPrimeField = false;
  static constexpr bool kIsPackedPrimeField = false;
  static constexpr bool kIsExtensionField = true;

  using Config = _Config;
};

template <typename _Config>
struct FiniteFieldTraits<Fp3<_Config>> {
  static constexpr bool kIsPrimeField = false;
  static constexpr bool kIsPackedPrimeField = false;
  static constexpr bool kIsExtensionField = true;

  using Config = _Config;
};

template <typename _Config>
struct FiniteFieldTraits<Fp4<_Config>> {
  static constexpr bool kIsPrimeField = false;
  static constexpr bool kIsPackedPrimeField = false;
  static constexpr bool kIsExtensionField = true;

  using Config = _Config;
};

template <typename _Config>
struct FiniteFieldTraits<Fp6<_Config>> {
  static constexpr bool kIsPrimeField = false;
  static constexpr bool kIsPackedPrimeField = false;
  static constexpr bool kIsExtensionField = true;

  using Config = _Config;
};

template <typename _Config>
struct FiniteFieldTraits<Fp12<_Config>> {
  static constexpr bool kIsPrimeField = false;
  static constexpr bool kIsPackedPrimeField = false;
  static constexpr bool kIsExtensionField = true;

  using Config = _Config;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_FINITE_FIELD_TRAITS_H_
