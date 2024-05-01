#ifndef TACHYON_MATH_MATRIX_PRIME_FIELD_NUM_TRAITS_H_
#define TACHYON_MATH_MATRIX_PRIME_FIELD_NUM_TRAITS_H_

#include <type_traits>

#include "third_party/eigen3/Eigen/Core"

#include "tachyon/math/finite_fields/finite_field_forwards.h"

namespace Eigen {

template <typename Config>
struct CostCalculator;

template <typename Config>
struct CostCalculator<tachyon::math::PrimeField<Config>> {
  constexpr static size_t kLimbNums =
      tachyon::math::PrimeField<Config>::kLimbNums;
  using NumTraitsType =
      std::conditional_t<Config::kModulusBits <= 32, NumTraits<uint32_t>,
                         NumTraits<uint64_t>>;

  constexpr static int ComputeReadCost() {
    return static_cast<int>(kLimbNums * NumTraitsType::ReadCost);
  }
  constexpr static int ComputeAddCost() {
    // In general, c = (a + b) % M = (a + b) > M ? (a + b) - M : (a + b)
    return static_cast<int>(kLimbNums * NumTraitsType::AddCost * 3 / 2);
  }
  constexpr static int ComputeMulCost() {
    // In general, c = (a * b) % M = (a * b) - [(a * b) / M] * M
    return static_cast<int>(
        kLimbNums * (4 * NumTraitsType::MulCost + NumTraitsType::AddCost));
  }
};

template <typename Config>
struct NumTraits<tachyon::math::PrimeField<Config>>
    : GenericNumTraits<tachyon::math::PrimeField<Config>> {
  enum {
    IsInteger = 1,
    IsSigned = 0,
    IsComplex = 0,
    RequireInitialization = 1,
    ReadCost =
        CostCalculator<tachyon::math::PrimeField<Config>>::ComputeReadCost(),
    AddCost =
        CostCalculator<tachyon::math::PrimeField<Config>>::ComputeAddCost(),
    MulCost =
        CostCalculator<tachyon::math::PrimeField<Config>>::ComputeMulCost(),
  };
};

}  // namespace Eigen

#endif  // TACHYON_MATH_MATRIX_PRIME_FIELD_NUM_TRAITS_H_
