#ifndef TACHYON_MATH_FINITE_FIELDS_KOALA_BEAR_INTERNAL_PACKED_KOALA_BEAR_H_
#define TACHYON_MATH_FINITE_FIELDS_KOALA_BEAR_INTERNAL_PACKED_KOALA_BEAR_H_

#include "tachyon/build/build_config.h"

#if ARCH_CPU_X86_64
#if defined(TACHYON_HAS_AVX512)
#include "tachyon/math/finite_fields/koala_bear/internal/packed_koala_bear_avx512.h"
#else
#include "tachyon/math/finite_fields/koala_bear/internal/packed_koala_bear_avx2.h"
#endif
#elif ARCH_CPU_ARM64
#include "tachyon/math/finite_fields/koala_bear/internal/packed_koala_bear_neon.h"
#endif
#include "tachyon/math/finite_fields/finite_field_traits.h"
#include "tachyon/math/matrix/prime_field_num_traits.h"

namespace tachyon::math {

#if ARCH_CPU_X86_64
#if defined(TACHYON_HAS_AVX512)
using PackedKoalaBear = PackedKoalaBearAVX512;
#else
using PackedKoalaBear = PackedKoalaBearAVX2;
#endif
#elif ARCH_CPU_ARM64
using PackedKoalaBear = PackedKoalaBearNeon;
#endif

template <>
struct FiniteFieldTraits<PackedKoalaBear> {
  static constexpr bool kIsPrimeField = true;
  static constexpr bool kIsPackedPrimeField = true;
  static constexpr bool kIsExtensionField = false;

  using PrimeField = KoalaBear;
  using Config = KoalaBear::Config;
};

}  // namespace tachyon::math

namespace Eigen {

template <>
struct NumTraits<tachyon::math::PackedKoalaBear>
    : GenericNumTraits<tachyon::math::PackedKoalaBear> {
  using PrimeField = tachyon::math::KoalaBear;
  constexpr static size_t N = tachyon::math::PackedKoalaBear::N;

  enum {
    IsInteger = 1,
    IsField = 1,
    IsSigned = 0,
    IsComplex = 0,
    RequireInitialization = 1,
    ReadCost = tachyon::math::CostCalculator<PrimeField>::ComputeReadCost() * N,
    AddCost = tachyon::math::CostCalculator<PrimeField>::ComputeAddCost() * N,
    MulCost = tachyon::math::CostCalculator<PrimeField>::ComputeMulCost() * N,
  };
};

}  // namespace Eigen

#endif  // TACHYON_MATH_FINITE_FIELDS_KOALA_BEAR_INTERNAL_PACKED_KOALA_BEAR_H_
