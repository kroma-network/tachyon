#ifndef TACHYON_MATH_FINITE_FIELDS_BABY_BEAR_PACKED_BABY_BEAR4_H_
#define TACHYON_MATH_FINITE_FIELDS_BABY_BEAR_PACKED_BABY_BEAR4_H_

#include "tachyon/math/finite_fields/baby_bear/internal/baby_bear4.h"
#include "tachyon/math/finite_fields/baby_bear/internal/packed_baby_bear4.h"
#include "tachyon/math/finite_fields/extended_packed_field_traits_forward.h"

namespace tachyon::math {

template <>
struct ExtendedPackedFieldTraits<PackedBabyBear4> {
  using PackedField = PackedBabyBear;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_BABY_BEAR_PACKED_BABY_BEAR4_H_
