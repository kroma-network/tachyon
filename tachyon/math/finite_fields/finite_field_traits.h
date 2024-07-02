#ifndef TACHYON_MATH_FINITE_FIELDS_FINITE_FIELD_TRAITS_H_
#define TACHYON_MATH_FINITE_FIELDS_FINITE_FIELD_TRAITS_H_

#include <stdint.h>

#include "third_party/eigen3/Eigen/Core"

#include "tachyon/math/finite_fields/finite_field_forwards.h"
#include "tachyon/math/matrix/cost_calculator_forward.h"

namespace tachyon::math {

template <typename T>
struct FiniteFieldTraits {
  static constexpr bool kIsFiniteField = false;
  static constexpr bool kIsPrimeField = false;
  static constexpr bool kIsPackedPrimeField = false;
  static constexpr bool kIsExtensionField = false;
};

template <typename _Config>
struct FiniteFieldTraits<BinaryField<_Config>> {
  static constexpr bool kIsFiniteField = true;
  static constexpr bool kIsPrimeField = false;
  static constexpr bool kIsPackedPrimeField = false;
  static constexpr bool kIsExtensionField = true;

  using Config = _Config;
};

template <typename _Config>
struct FiniteFieldTraits<PrimeField<_Config>> {
  static constexpr bool kIsFiniteField = true;
  static constexpr bool kIsPrimeField = true;
  static constexpr bool kIsPackedPrimeField = false;
  static constexpr bool kIsExtensionField = false;

  using PrimeField = tachyon::math::PrimeField<_Config>;
  using Config = _Config;
};

template <typename _Config>
struct FiniteFieldTraits<PrimeFieldGpu<_Config>> {
  static constexpr bool kIsFiniteField = true;
  static constexpr bool kIsPrimeField = true;
  static constexpr bool kIsPackedPrimeField = false;
  static constexpr bool kIsExtensionField = false;

  using Config = _Config;
};

template <typename _Config>
struct FiniteFieldTraits<PrimeFieldGpuDebug<_Config>> {
  static constexpr bool kIsFiniteField = true;
  static constexpr bool kIsPrimeField = true;
  static constexpr bool kIsPackedPrimeField = false;
  static constexpr bool kIsExtensionField = false;

  using Config = _Config;
};

template <typename _Config>
struct FiniteFieldTraits<Fp2<_Config>> {
  static constexpr bool kIsFiniteField = true;
  static constexpr bool kIsPrimeField = false;
  static constexpr bool kIsPackedPrimeField = false;
  static constexpr bool kIsExtensionField = true;

  using Config = _Config;
};

template <typename _Config>
struct FiniteFieldTraits<Fp3<_Config>> {
  static constexpr bool kIsFiniteField = true;
  static constexpr bool kIsPrimeField = false;
  static constexpr bool kIsPackedPrimeField = false;
  static constexpr bool kIsExtensionField = true;

  using Config = _Config;
};

template <typename _Config>
struct FiniteFieldTraits<Fp4<_Config>> {
  static constexpr bool kIsFiniteField = true;
  static constexpr bool kIsPrimeField = false;
  static constexpr bool kIsPackedPrimeField = false;
  static constexpr bool kIsExtensionField = true;

  using Config = _Config;
};

template <typename _Config>
struct FiniteFieldTraits<Fp6<_Config>> {
  static constexpr bool kIsFiniteField = true;
  static constexpr bool kIsPrimeField = false;
  static constexpr bool kIsPackedPrimeField = false;
  static constexpr bool kIsExtensionField = true;

  using Config = _Config;
};

template <typename _Config>
struct FiniteFieldTraits<Fp12<_Config>> {
  static constexpr bool kIsFiniteField = true;
  static constexpr bool kIsPrimeField = false;
  static constexpr bool kIsPackedPrimeField = false;
  static constexpr bool kIsExtensionField = true;

  using Config = _Config;
};

}  // namespace tachyon::math

namespace Eigen {

template <typename F>
struct NumTraits<
    F, std::enable_if_t<tachyon::math::FiniteFieldTraits<F>::kIsExtensionField>>
    : GenericNumTraits<F> {
  using BasePrimeField = typename F::BasePrimeField;
  constexpr static uint32_t kDegreeOverBasePrimeField =
      F::kDegreeOverBasePrimeField;

  enum {
    IsInteger = 1,
    IsSigned = 0,
    IsComplex = 0,
    RequireInitialization = 1,
    ReadCost =
        tachyon::math::CostCalculator<BasePrimeField>::ComputeReadCost() *
        kDegreeOverBasePrimeField,
    AddCost = tachyon::math::CostCalculator<BasePrimeField>::ComputeAddCost() *
              kDegreeOverBasePrimeField,
    MulCost = tachyon::math::CostCalculator<BasePrimeField>::ComputeMulCost() *
              kDegreeOverBasePrimeField,
  };
};

}  // namespace Eigen

#endif  // TACHYON_MATH_FINITE_FIELDS_FINITE_FIELD_TRAITS_H_
