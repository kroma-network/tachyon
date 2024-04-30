#ifndef TACHYON_MATH_FINITE_FIELDS_GOLDILOCKS_GOLDILOCKS_PRIME_FIELD_H_
#define TACHYON_MATH_FINITE_FIELDS_GOLDILOCKS_GOLDILOCKS_PRIME_FIELD_H_

#include "tachyon/build/build_config.h"

#if defined(TACHYON_HAS_ASM_PRIME_FIELD) && ARCH_CPU_X86_64
#include "tachyon/math/finite_fields/goldilocks/goldilocks_prime_field_x86_special.h"
#else
#include "tachyon/math/finite_fields/goldilocks/goldilocks.h"
#endif

namespace tachyon::math {

using Goldilocks = PrimeField<GoldilocksConfig>;

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_GOLDILOCKS_GOLDILOCKS_PRIME_FIELD_H_
