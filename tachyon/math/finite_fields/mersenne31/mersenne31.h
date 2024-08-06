#ifndef TACHYON_MATH_FINITE_FIELDS_MERSENNE31_MERSENNE31_H_
#define TACHYON_MATH_FINITE_FIELDS_MERSENNE31_MERSENNE31_H_

#include "tachyon/math/finite_fields/mersenne31/internal/packed_mersenne31.h"

namespace tachyon::math {

template <>
struct PackedFieldTraits<Mersenne31> {
  using PackedField = PackedMersenne31;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_MERSENNE31_MERSENNE31_H_
