#ifndef TACHYON_MATH_FINITE_FIELDS_KOALA_BEAR_PACKED_KOALA_BEAR4_H_
#define TACHYON_MATH_FINITE_FIELDS_KOALA_BEAR_PACKED_KOALA_BEAR4_H_

#include "tachyon/math/finite_fields/baby_bear/internal/koala_bear4.h"
#include "tachyon/math/finite_fields/baby_bear/internal/packed_koala_bear4.h"
#include "tachyon/math/finite_fields/extended_packed_field_traits_forward.h"

namespace tachyon::math {

template <>
struct ExtendedPackedFieldTraits<PackedKoalaBear4> {
  using PackedField = PackedKoalaBear;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_KOALA_BEAR_PACKED_KOALA_BEAR4_H_
