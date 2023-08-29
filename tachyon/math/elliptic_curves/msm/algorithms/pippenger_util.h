#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_PIPPENGER_UTIL_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_PIPPENGER_UTIL_H_

#include <cmath>

namespace tachyon::math {

// The result of this function is only approximately `ln(a)`.
// See https://github.com/scipr-lab/zexe/issues/79#issue-556220473
constexpr size_t LnWithoutFloats(size_t a) {
  // log2(a) * ln(2)
  return log2(a) * 69 / 100;
}

constexpr size_t ComputeWindowsBits(size_t size) {
  if (size < 32) {
    return 3;
  } else {
    return LnWithoutFloats(size) + 2;
  }
}

template <typename ScalarField>
constexpr size_t ComputeWindowsCount(size_t window_bits) {
  return (ScalarField::Config::kModulusBits + window_bits - 1) / window_bits;
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_PIPPENGER_UTIL_H_
