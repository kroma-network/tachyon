#ifndef TACHYON_MATH_FINITE_FIELDS_BABY_BEAR_BABY_BEAR_H_
#define TACHYON_MATH_FINITE_FIELDS_BABY_BEAR_BABY_BEAR_H_

#include "tachyon/math/finite_fields/baby_bear/internal/packed_baby_bear.h"

namespace tachyon::math {

template <>
struct PackedFieldTraits<BabyBear> {
  using PackedField = PackedBabyBear;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_BABY_BEAR_BABY_BEAR_H_
