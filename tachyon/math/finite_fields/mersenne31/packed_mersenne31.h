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

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_MERSENNE31_PACKED_MERSENNE31_H_
