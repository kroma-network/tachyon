#ifndef TACHYON_MATH_MATRIX_PRIME_FIELD_NUM_TRAITS_H_
#define TACHYON_MATH_MATRIX_PRIME_FIELD_NUM_TRAITS_H_

#include "third_party/eigen3/Eigen/Core"

#include "tachyon/math/finite_fields/finite_field_forwards.h"

namespace Eigen {

template <typename Config>
struct NumTraits<tachyon::math::PrimeField<Config>>
    : GenericNumTraits<tachyon::math::PrimeField<Config>> {
  constexpr static size_t kLimbNums =
      tachyon::math::PrimeField<Config>::kLimbNums;

  enum {
    IsInteger = 1,
    IsSigned = 0,
    IsComplex = 0,
    RequireInitialization = 1,
    ReadCost = static_cast<int>(kLimbNums * NumTraits<uint64_t>::ReadCost),
    // In general, c = (a + b) % M = (a + b) > M ? (a + b) - M : (a + b)
    AddCost =
        static_cast<int>(kLimbNums * NumTraits<uint64_t>::AddCost * 3 / 2),
    // In general, c = (a * b) % M = (a * b) - [(a * b) / M] * M
    MulCost = static_cast<int>(kLimbNums * (4 * NumTraits<uint64_t>::MulCost +
                                            NumTraits<uint64_t>::AddCost)),
  };
};

#if defined(TACHYON_GMP_BACKEND)
template <typename Config>
struct NumTraits<tachyon::math::PrimeFieldGmp<Config>>
    : GenericNumTraits<tachyon::math::PrimeFieldGmp<Config>> {
  constexpr static size_t kLimbNums =
      tachyon::math::PrimeFieldGmp<Config>::kLimbNums;

  enum {
    IsInteger = 1,
    IsSigned = 0,
    IsComplex = 0,
    RequireInitialization = 1,
    ReadCost = static_cast<int>(kLimbNums * NumTraits<uint64_t>::ReadCost),
    // In general, c = (a + b) % M = (a + b) > M ? (a + b) - M : (a + b)
    AddCost =
        static_cast<int>(kLimbNums * NumTraits<uint64_t>::AddCost * 3 / 2),
    // In general, c = (a * b) % M = (a * b) - [(a * b) / M] * M
    MulCost = static_cast<int>(kLimbNums * (4 * NumTraits<uint64_t>::MulCost +
                                            NumTraits<uint64_t>::AddCost)),
  };
};
#endif  // defined(TACHYON_GMP_BACKEND)

}  // namespace Eigen

#endif  // TACHYON_MATH_MATRIX_PRIME_FIELD_NUM_TRAITS_H_
