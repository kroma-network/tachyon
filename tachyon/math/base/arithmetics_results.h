#ifndef TACHYON_MATH_BASE_ARITHMETICS_RESULTS_H_
#define TACHYON_MATH_BASE_ARITHMETICS_RESULTS_H_

#include <string>

#include "absl/strings/substitute.h"

namespace tachyon::math {

template <typename T>
struct AddResult {
  T result{};
  T carry{};

  constexpr bool operator==(const AddResult& other) const {
    return result == other.result && carry == other.carry;
  }
  constexpr bool operator!=(const AddResult& other) const {
    return !operator==(other);
  }

  std::string ToString() const {
    return absl::Substitute("($0, $1)", result.ToString(), carry.ToString());
  }
};

template <typename T>
struct SubResult {
  T result{};
  T borrow{};

  constexpr bool operator==(const SubResult& other) const {
    return result == other.result && borrow == other.borrow;
  }
  constexpr bool operator!=(const SubResult& other) const {
    return !operator==(other);
  }

  std::string ToString() const {
    return absl::Substitute("($0, $1)", result.ToString(), borrow.ToString());
  }
};

template <typename T>
struct MulResult {
  T hi{};
  T lo{};

  constexpr bool operator==(const MulResult& other) const {
    return hi == other.hi && lo == other.lo;
  }
  constexpr bool operator!=(const MulResult& other) const {
    return !operator==(other);
  }

  std::string ToString() const {
    return absl::Substitute("($0, $1)", hi.ToString(), lo.ToString());
  }
};

template <typename T>
struct DivResult {
  T quotient{};
  T remainder{};

  constexpr bool operator==(const DivResult& other) const {
    return quotient == other.quotient && remainder == other.remainder;
  }
  constexpr bool operator!=(const DivResult& other) const {
    return !operator==(other);
  }

  std::string ToString() const {
    return absl::Substitute("($0, $1)", quotient.ToString(),
                            remainder.ToString());
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_BASE_ARITHMETICS_RESULTS_H_
