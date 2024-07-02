#ifndef TACHYON_MATH_FINITE_FIELDS_MERSENNE31_PACKED_MERSENNE31_H_
#define TACHYON_MATH_FINITE_FIELDS_MERSENNE31_PACKED_MERSENNE31_H_

#include "tachyon/build/build_config.h"

#if ARCH_CPU_X86_64
#if defined(TACHYON_HAS_AVX512)
#include "tachyon/math/finite_fields/mersenne31/packed_mersenne31_avx512.h"
#else
#include "tachyon/math/finite_fields/mersenne31/packed_mersenne31_avx2.h"
#endif
#elif ARCH_CPU_ARM64
#include "tachyon/math/finite_fields/mersenne31/packed_mersenne31_neon.h"
#endif
#include "tachyon/math/finite_fields/finite_field_traits.h"
#include "tachyon/math/matrix/prime_field_num_traits.h"

namespace tachyon::math {

#if ARCH_CPU_X86_64
#if defined(TACHYON_HAS_AVX512)
using PackedMersenne31 = PackedMersenne31AVX512;
#else
using PackedMersenne31 = PackedMersenne31AVX2;
#endif
#elif ARCH_CPU_ARM64
using PackedMersenne31 = PackedMersenne31Neon;
#endif

template <>
struct FiniteFieldTraits<PackedMersenne31> {
  static constexpr bool kIsPrimeField = true;
  static constexpr bool kIsPackedPrimeField = true;
  static constexpr bool kIsExtensionField = false;

  using PrimeField = Mersenne31;
  using Config = Mersenne31::Config;
};

template <>
struct PackedPrimeFieldTraits<Mersenne31> {
  using PackedPrimeField = PackedMersenne31;
};

}  // namespace tachyon::math

namespace Eigen {

template <>
struct NumTraits<tachyon::math::PackedMersenne31>
    : GenericNumTraits<tachyon::math::PackedMersenne31> {
  using PrimeField = tachyon::math::Mersenne31;
  constexpr static size_t N = tachyon::math::PackedMersenne31::N;

  enum {
    IsInteger = 1,
    IsSigned = 0,
    IsComplex = 0,
    RequireInitialization = 1,
    ReadCost = tachyon::math::CostCalculator<PrimeField>::ComputeReadCost() * N,
    AddCost = tachyon::math::CostCalculator<PrimeField>::ComputeAddCost() * N,
    MulCost = tachyon::math::CostCalculator<PrimeField>::ComputeMulCost() * N,
  };
};

}  // namespace Eigen

#endif  // TACHYON_MATH_FINITE_FIELDS_MERSENNE31_PACKED_MERSENNE31_H_
