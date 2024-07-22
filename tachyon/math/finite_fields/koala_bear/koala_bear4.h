#ifndef TACHYON_MATH_FINITE_FIELDS_KOALA_BEAR_KOALA_BEAR4_H_
#define TACHYON_MATH_FINITE_FIELDS_KOALA_BEAR_KOALA_BEAR4_H_

#include "tachyon/math/finite_fields/baby_bear/internal/koala_bear.h"
#include "tachyon/math/finite_fields/baby_bear/internal/packed_koala_bear.h"
#include "tachyon/math/finite_fields/extended_packed_field_traits_forward.h"

namespace tachyon::math {

template <>
struct ExtendedPackedFieldTraits<KoalaBear4> {
  using ExtendedPackedField = PackedKoalaBear4;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_KOALA_BEAR_KOALA_BEAR4_H_
