#ifndef TACHYON_MATH_FINITE_FIELDS_MERSENNE31_POSEIDON2_H_
#define TACHYON_MATH_FINITE_FIELDS_MERSENNE31_POSEIDON2_H_

#include <array>

#include "tachyon/base/types/always_false.h"
#include "tachyon/math/finite_fields/mersenne31/mersenne31.h"

namespace tachyon::math {

template <size_t N>
std::array<uint8_t, N> GetPoseidon2Mersenne31InternalShiftVector() {
  // This is taken and modified from
  // https://github.com/Plonky3/Plonky3/blob/fde81db/mersenne-31/src/poseidon2.rs.
  if constexpr (N == 15) {
    return {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16};
  } else if constexpr (N == 23) {
    return {
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    };
  } else {
    static_assert(base::AlwaysFalse<std::array<uint8_t, N>>);
  }
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_MERSENNE31_POSEIDON2_H_
