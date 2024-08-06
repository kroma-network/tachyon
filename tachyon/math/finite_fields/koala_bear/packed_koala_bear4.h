#ifndef TACHYON_MATH_FINITE_FIELDS_KOALA_BEAR_PACKED_KOALA_BEAR4_H_
#define TACHYON_MATH_FINITE_FIELDS_KOALA_BEAR_PACKED_KOALA_BEAR4_H_

#include "tachyon/math/finite_fields/baby_bear/internal/packed_koala_bear4.h"

namespace tachyon::math {

template <>
struct ExtendedPackedFieldTraits<PackedKoalaBear4> {
  constexpr static bool kIsExtendedPackedField = true;
  // Note(ashjeong): |ExtendedPackedField| is defaulted as itself. See
  // extension_field_base.h to see how it is used.
  using ExtendedPackedField = PackedKoalaBear4;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_KOALA_BEAR_PACKED_KOALA_BEAR4_H_
