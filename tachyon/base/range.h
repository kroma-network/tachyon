#ifndef TACHYON_BASE_RANGE_H_
#define TACHYON_BASE_RANGE_H_

#include <limits>
#include <string>
#include <type_traits>

#include "absl/strings/substitute.h"

namespace tachyon::base {

template <typename T, bool IsStartInclusive = true, bool IsEndInclusive = false,
          typename SFINAE = void>
struct Range;

template <typename T, bool IsStartInclusive, bool IsEndInclusive>
struct Range<T, IsStartInclusive, IsEndInclusive,
             std::enable_if_t<std::is_arithmetic_v<T>>> {
  constexpr static bool kIsStartInclusive = IsStartInclusive;
  constexpr static bool kIsEndInclusive = IsEndInclusive;

  // Returns the lowest finite value representable by the numeric type T, that
  // is, a finite value x such that there is no other finite value y where
  // y < x. This is different from std::numeric_limits<T>::min() for
  // floating-point types. Only meaningful for bounded types.
  // See https://en.cppreference.com/w/cpp/types/numeric_limits/lowest
  // NOTE(chokobole): I used |lowest()| over |min()| for the reason above.
  T start = std::numeric_limits<T>::lowest();
  T end = std::numeric_limits<T>::max();

  constexpr Range() = default;
  constexpr Range(T start, T end) : start(start), end(end) {}

  constexpr static Range All() { return Range(); }

  constexpr static Range From(T start) {
    Range range;
    range.start = start;
    return range;
  }

  constexpr static Range Until(T end) {
    Range range;
    range.end = end;
    return range;
  }

  // Returns true if the range doesn't contain any value. |Contains()| always
  // gives you false in this case.
  constexpr bool IsEmpty() const {
    if constexpr (IsStartInclusive && IsEndInclusive) {
      return start > end;
    } else {
      return start >= end;
    }
  }

  // Returns true if the range contains |value|.
  constexpr bool Contains(T value) const {
    if constexpr (IsStartInclusive && IsEndInclusive) {
      return start <= value && value <= end;
    } else if constexpr (IsStartInclusive && !IsEndInclusive) {
      return start <= value && value < end;
    } else if constexpr (!IsStartInclusive && IsEndInclusive) {
      return start < value && value <= end;
    } else {
      return start < value && value < end;
    }
  }

  std::string ToString() const {
    if constexpr (IsStartInclusive && IsEndInclusive) {
      return absl::Substitute("$0 ≤ x ≤ $1", start, end);
    } else if constexpr (IsStartInclusive && !IsEndInclusive) {
      return absl::Substitute("$0 ≤ x < $1", start, end);
    } else if constexpr (!IsStartInclusive && IsEndInclusive) {
      return absl::Substitute("$0 < x ≤ $1", start, end);
    } else {
      return absl::Substitute("$0 < x < $1", start, end);
    }
  }
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_RANGE_H_
