#ifndef TACHYON_MATH_BASE_LAGRANGE_SELECTORS_H_
#define TACHYON_MATH_BASE_LAGRANGE_SELECTORS_H_

namespace tachyon::math {

template <typename T>
struct LagrangeSelectors {
  T first_row;
  T last_row;
  T transition;
  T inv_zeroifier;

  bool operator==(const LagrangeSelectors& other) const {
    return first_row == other.first_row && last_row == other.last_row &&
           transition == other.transition &&
           inv_zeroifier == other.inv_zeroifier;
  }
  bool operator!=(const LagrangeSelectors& other) const {
    return !operator==(other);
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_BASE_LAGRANGE_SELECTORS_H_
