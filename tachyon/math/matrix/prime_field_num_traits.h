#ifndef TACHYON_MATH_MATRIX_PRIME_FIELD_NUM_TRAITS_H_
#define TACHYON_MATH_MATRIX_PRIME_FIELD_NUM_TRAITS_H_

#include <type_traits>

#include "third_party/eigen3/Eigen/Core"

#include "tachyon/math/finite_fields/finite_field_forwards.h"
#include "tachyon/math/matrix/cost_calculator_forward.h"

namespace tachyon::math {

template <typename Config>
struct CostCalculator<PrimeField<Config>> {
  constexpr static size_t kLimbNums = PrimeField<Config>::kLimbNums;
  using NumTraitsType =
      std::conditional_t<Config::kModulusBits <= 32, Eigen::NumTraits<uint32_t>,
                         Eigen::NumTraits<uint64_t>>;

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

}  // namespace tachyon::math

namespace Eigen {

template <typename Config>
struct NumTraits<tachyon::math::PrimeField<Config>>
    : GenericNumTraits<tachyon::math::PrimeField<Config>> {
  enum {
    IsInteger = 1,
    IsSigned = 0,
    IsComplex = 0,
    RequireInitialization = 1,
    ReadCost = tachyon::math::CostCalculator<
        tachyon::math::PrimeField<Config>>::ComputeReadCost(),
    AddCost = tachyon::math::CostCalculator<
        tachyon::math::PrimeField<Config>>::ComputeAddCost(),
    MulCost = tachyon::math::CostCalculator<
        tachyon::math::PrimeField<Config>>::ComputeMulCost(),
  };
};

}  // namespace Eigen

#endif  // TACHYON_MATH_MATRIX_PRIME_FIELD_NUM_TRAITS_H_
