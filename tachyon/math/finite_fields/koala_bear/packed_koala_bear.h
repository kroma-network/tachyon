#ifndef TACHYON_MATH_FINITE_FIELDS_KOALA_BEAR_PACKED_KOALA_BEAR_H_
#define TACHYON_MATH_FINITE_FIELDS_KOALA_BEAR_PACKED_KOALA_BEAR_H_

#include "tachyon/build/build_config.h"

#if ARCH_CPU_X86_64
#if defined(TACHYON_HAS_AVX512)
#include "tachyon/math/finite_fields/koala_bear/packed_koala_bear_avx512.h"
#else
#include "tachyon/math/finite_fields/koala_bear/packed_koala_bear_avx2.h"
#endif
#elif ARCH_CPU_ARM64
#include "tachyon/math/finite_fields/koala_bear/packed_koala_bear_neon.h"
#endif

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

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_KOALA_BEAR_PACKED_KOALA_BEAR_H_
