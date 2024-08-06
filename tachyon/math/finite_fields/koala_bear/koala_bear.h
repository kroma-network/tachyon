#ifndef TACHYON_MATH_FINITE_FIELDS_KOALA_BEAR_KOALA_BEAR_H_
#define TACHYON_MATH_FINITE_FIELDS_KOALA_BEAR_KOALA_BEAR_H_

#include "tachyon/math/finite_fields/koala_bear/internal/packed_koala_bear.h"

namespace tachyon::math {

template <>
struct PackedFieldTraits<KoalaBear> {
  using PackedField = PackedKoalaBear;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_KOALA_BEAR_KOALA_BEAR_H_
