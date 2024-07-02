#ifndef TACHYON_MATH_MATRIX_PRIME_FIELD_NUM_TRAITS_H_
#define TACHYON_MATH_MATRIX_PRIME_FIELD_NUM_TRAITS_H_

#include <type_traits>

#include "third_party/eigen3/Eigen/Core"

#include "tachyon/math/finite_fields/prime_field_base.h"
#include "tachyon/math/matrix/cost_calculator_forward.h"

namespace tachyon::math {

template <typename F>
struct CostCalculator<
    F, std::enable_if_t<std::is_base_of_v<PrimeFieldBase<F>, F>>> {
  constexpr static size_t kLimbNums = F::kLimbNums;
  // NOLINTNEXTLINE(whitespace/operators)
  using NumTraitsType = std::conditional_t<F::Config::kModulusBits <= 32,
                                           Eigen::NumTraits<uint32_t>,
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

template <typename F>
struct NumTraits<
    F, std::enable_if_t<std::is_base_of_v<tachyon::math::PrimeFieldBase<F>, F>>>
    : GenericNumTraits<F> {
  enum {
    IsInteger = 1,
    IsSigned = 0,
    IsComplex = 0,
    RequireInitialization = 1,
    ReadCost = tachyon::math::CostCalculator<F>::ComputeReadCost(),
    AddCost = tachyon::math::CostCalculator<F>::ComputeAddCost(),
    MulCost = tachyon::math::CostCalculator<F>::ComputeMulCost(),
  };
};

}  // namespace Eigen

#endif  // TACHYON_MATH_MATRIX_PRIME_FIELD_NUM_TRAITS_H_
