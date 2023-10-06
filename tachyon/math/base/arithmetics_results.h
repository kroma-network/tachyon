#ifndef TACHYON_MATH_BASE_ARITHMETICS_RESULTS_H_
#define TACHYON_MATH_BASE_ARITHMETICS_RESULTS_H_

#include <ostream>
#include <string>

#include "absl/strings/substitute.h"

namespace tachyon::math {

template <typename T>
struct AddResult {
  T result = 0;
  T carry = 0;

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
std::ostream& operator<<(std::ostream& os, const AddResult<T>& result) {
  return os << result.ToString();
}

template <typename T>
struct SubResult {
  T result = 0;
  T borrow = 0;

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
std::ostream& operator<<(std::ostream& os, const SubResult<T>& result) {
  return os << result.ToString();
}

template <typename T>
struct MulResult {
  T hi = 0;
  T lo = 0;

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
std::ostream& operator<<(std::ostream& os, const MulResult<T>& result) {
  return os << result.ToString();
}

template <typename T>
struct DivResult {
  T quotient = 0;
  T remainder = 0;

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

template <typename T>
std::ostream& operator<<(std::ostream& os, const DivResult<T>& result) {
  return os << result.ToString();
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_BASE_ARITHMETICS_RESULTS_H_
