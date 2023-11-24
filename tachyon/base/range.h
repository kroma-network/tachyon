#ifndef TACHYON_BASE_RANGE_H_
#define TACHYON_BASE_RANGE_H_

#include <algorithm>
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

  class Iterator {
   public:
    static_assert(std::is_integral_v<T>);

    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = value_type*;
    using reference = value_type&;
    using iterator_category = std::random_access_iterator_tag;

    constexpr Iterator() = default;
    constexpr explicit Iterator(T cur) : cur_(cur) {}

    constexpr bool operator==(Iterator other) const {
      return cur_ == other.cur_;
    }

    constexpr bool operator!=(Iterator other) const {
      return cur_ != other.cur_;
    }

    constexpr bool operator<(Iterator other) const { return cur_ < other.cur_; }

    constexpr bool operator<=(Iterator other) const {
      return cur_ <= other.cur_;
    }

    constexpr bool operator>(Iterator other) const { return cur_ > other.cur_; }

    constexpr bool operator>=(Iterator other) const {
      return cur_ >= other.cur_;
    }

    constexpr Iterator& operator++() {
      ++cur_;
      return *this;
    }

    constexpr Iterator operator++(int) const {
      Iterator it(*this);
      ++(*this);
      return it;
    }

    constexpr Iterator& operator--() {
      --cur_;
      return *this;
    }

    constexpr Iterator operator--(int) const {
      Iterator it(*this);
      --(*this);
      return it;
    }

    constexpr Iterator operator+(difference_type n) const {
      return Iterator(cur_ + n);
    }

    constexpr Iterator& operator+=(difference_type n) {
      cur_ += n;
      return *this;
    }

    constexpr Iterator operator-(difference_type n) const {
      return Iterator(cur_ - n);
    }

    constexpr Iterator& operator-=(difference_type n) {
      cur_ -= n;
      return *this;
    }

    constexpr difference_type operator-(Iterator other) const {
      return cur_ - other.cur_;
    }

    constexpr reference operator*() { return cur_; }

    constexpr pointer operator->() { return &cur_; }

   private:
    T cur_ = 0;
  };

  // Returns the lowest finite value representable by the numeric type T, that
  // is, a finite value x such that there is no other finite value y where
  // y < x. This is different from std::numeric_limits<T>::min() for
  // floating-point types. Only meaningful for bounded types.
  // See https://en.cppreference.com/w/cpp/types/numeric_limits/lowest
  // NOTE(chokobole): I used |lowest()| over |min()| for the reason above.
  T from = std::numeric_limits<T>::lowest();
  T to = std::numeric_limits<T>::max();

  constexpr Range() = default;
  constexpr Range(T from, T to) : from(from), to(to) {}

  constexpr static Range All() { return Range(); }

  constexpr static Range From(T from) {
    Range range;
    range.from = from;
    return range;
  }

  constexpr static Range Until(T to) {
    Range range;
    range.to = to;
    return range;
  }

  Iterator begin() const {
    if constexpr (IsStartInclusive) {
      return Iterator(from);
    } else {
      return Iterator(from + 1);
    }
  }

  Iterator end() const {
    if constexpr (IsEndInclusive) {
      return Iterator(to + 1);
    } else {
      return Iterator(to);
    }
  }

  constexpr Range Intersect(Range other) const {
    return Range(std::max(from, other.from), std::min(to, other.to));
  }

  // Returns the number of integral elements within the range.
  template <typename U = T, std::enable_if_t<std::is_integral_v<U>>* = nullptr>
  constexpr size_t GetSize() const {
    if (IsEmpty()) return 0;
    if constexpr (IsStartInclusive && IsEndInclusive) {
      return to - from + 1;
    } else if constexpr (IsStartInclusive && !IsEndInclusive) {
      return to - from;
    } else if constexpr (!IsStartInclusive && IsEndInclusive) {
      return to - from;
    } else {
      return to - from - 1;
    }
  }

  // Returns true if the range doesn't contain any value. |Contains()| always
  // gives you false in this case.
  constexpr bool IsEmpty() const {
    if constexpr (IsStartInclusive && IsEndInclusive) {
      return from > to;
    } else {
      return from >= to;
    }
  }

  // Returns true if the range contains |value|.
  constexpr bool Contains(T value) const {
    if constexpr (IsStartInclusive && IsEndInclusive) {
      return from <= value && value <= to;
    } else if constexpr (IsStartInclusive && !IsEndInclusive) {
      return from <= value && value < to;
    } else if constexpr (!IsStartInclusive && IsEndInclusive) {
      return from < value && value <= to;
    } else {
      return from < value && value < to;
    }
  }

  constexpr bool operator==(Range other) const {
    return from == other.from && to == other.to;
  }
  constexpr bool operator!=(Range other) const { return !operator==(other); }

  std::string ToString() const {
    if constexpr (IsStartInclusive && IsEndInclusive) {
      return absl::Substitute("$0 ≤ x ≤ $1", from, to);
    } else if constexpr (IsStartInclusive && !IsEndInclusive) {
      return absl::Substitute("$0 ≤ x < $1", from, to);
    } else if constexpr (!IsStartInclusive && IsEndInclusive) {
      return absl::Substitute("$0 < x ≤ $1", from, to);
    } else {
      return absl::Substitute("$0 < x < $1", from, to);
    }
  }
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_RANGE_H_
