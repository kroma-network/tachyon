#ifndef TACHYON_BASE_TYPES_CXX20_IS_BOUNDED_ARRAY_H_
#define TACHYON_BASE_TYPES_CXX20_IS_BOUNDED_ARRAY_H_

#include <stddef.h>

#include <type_traits>

namespace tachyon::base {

// Implementation of C++20's std::is_bounded_array.
//
// References:
// - https://en.cppreference.com/w/cpp/types/is_bounded_array
template <typename T>
struct is_bounded_array : std::false_type {};

template <typename T, size_t N>
struct is_bounded_array<T[N]> : std::true_type {};

template <typename T>
inline constexpr bool is_bounded_array_v = is_bounded_array<T>::value;

}  // namespace tachyon::base

#endif  // TACHYON_BASE_TYPES_CXX20_IS_BOUNDED_ARRAY_H_
