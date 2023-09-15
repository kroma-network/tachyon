// Copyright (c) 2011 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// clang-format off

#ifndef TACHYON_BASE_TEMPLATE_UTIL_H_
#define TACHYON_BASE_TEMPLATE_UTIL_H_

#include <stddef.h>

#include <iterator>
#include <type_traits>

namespace tachyon::base {
namespace internal {

// Used to detect whether the given type is an iterator.  This is normally used
// with std::enable_if to provide disambiguation for functions that take
// templated iterators as input.
template <typename T, typename = void>
struct is_iterator : std::false_type {};

template <typename T>
struct is_iterator<
    T,
    std::void_t<typename std::iterator_traits<T>::iterator_category>>
    : std::true_type {};

// Helper to express preferences in an overload set. If more than one overload
// are available for a given set of parameters the overload with the higher
// priority will be chosen.
template <size_t I>
struct priority_tag : priority_tag<I - 1> {};

template <>
struct priority_tag<0> {};

}  // namespace internal

// Implementation of C++20's std::remove_cvref.
//
// References:
// - https://en.cppreference.com/w/cpp/types/remove_cvref
// - https://wg21.link/meta.trans.other#lib:remove_cvref
template <typename T>
struct remove_cvref {
  using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

// Implementation of C++20's std::remove_cvref_t.
//
// References:
// - https://en.cppreference.com/w/cpp/types/remove_cvref
// - https://wg21.link/meta.type.synop#lib:remove_cvref_t
template <typename T>
using remove_cvref_t = typename remove_cvref<T>::type;

// Simplified implementation of C++20's std::iter_value_t.
// As opposed to std::iter_value_t, this implementation does not restrict
// the type of `Iter` and does not consider specializations of
// `indirectly_readable_traits`.
//
// Reference: https://wg21.link/readable.traits#2
template <typename Iter>
using iter_value_t =
    typename std::iterator_traits<remove_cvref_t<Iter>>::value_type;

// Simplified implementation of C++20's std::iter_reference_t.
// As opposed to std::iter_reference_t, this implementation does not restrict
// the type of `Iter`.
//
// Reference: https://wg21.link/iterator.synopsis#:~:text=iter_reference_t
template <typename Iter>
using iter_reference_t = decltype(*std::declval<Iter&>());

// Simplified implementation of C++20's std::indirect_result_t. As opposed to
// std::indirect_result_t, this implementation does not restrict the type of
// `Func` and `Iters`.
//
// Reference: https://wg21.link/iterator.synopsis#:~:text=indirect_result_t
template <typename Func, typename... Iters>
using indirect_result_t =
    std::invoke_result_t<Func, iter_reference_t<Iters>...>;

// Simplified implementation of C++20's std::projected. As opposed to
// std::projected, this implementation does not explicitly restrict the type of
// `Iter` and `Proj`, but rather does so implicitly by requiring
// `indirect_result_t<Proj, Iter>` is a valid type. This is required for SFINAE
// friendliness.
//
// Reference: https://wg21.link/projected
template <typename Iter, typename Proj,
          typename IndirectResultT = indirect_result_t<Proj, Iter>>
struct projected {
  using value_type = remove_cvref_t<IndirectResultT>;

  IndirectResultT operator*() const;  // not defined
};

// Taken from
// https://github.com/pybind/pybind11/blob/master/include/pybind11/detail/common.h
// -------------------------------------------------------------------------------
/// Like is_base_of, but requires a strict base (i.e. `is_strict_base_of<T,
/// T>::value == false`, unlike `std::is_base_of`)
template <typename Base, typename Derived>
using is_strict_base_of = std::bool_constant<std::is_base_of<Base, Derived>::value &&
                                        !std::is_same<Base, Derived>::value>;

/// Compile-time all/any/none of that check the boolean value of all template
/// types
#if defined(__cpp_fold_expressions) && !(defined(_MSC_VER) && (_MSC_VER < 1916))
template <class... Ts>
using all_of = std::bool_constant<(Ts::value && ...)>;
template <class... Ts>
using any_of = std::bool_constant<(Ts::value || ...)>;
#elif !defined(_MSC_VER)
template <bool...>
struct bools {};
template <class... Ts>
using all_of =
    std::is_same<bools<Ts::value..., true>, bools<true, Ts::value...>>;
template <class... Ts>
using any_of = std::negation<all_of<std::negation<Ts>...>>;
#else
// MSVC has trouble with the above, but supports std::conjunction, which we can
// use instead (albeit at a slight loss of compilation efficiency).
template <class... Ts>
using all_of = std::conjunction<Ts...>;
template <class... Ts>
using any_of = std::disjunction<Ts...>;
#endif
template <class... Ts>
using none_of = std::negation<any_of<Ts...>>;

/// Compile-time integer sum
#ifdef __cpp_fold_expressions
template <typename... Ts>
constexpr size_t constexpr_sum(Ts... ns) {
  return (0 + ... + size_t{ns});
}
#else
constexpr size_t constexpr_sum() { return 0; }
template <typename T, typename... Ts>
constexpr size_t constexpr_sum(T n, Ts... ns) {
  return size_t{n} + constexpr_sum(ns...);
}
#endif

/// Implementation details for constexpr functions
constexpr int first(int i) { return i; }
template <typename T, typename... Ts>
constexpr int first(int i, T v, Ts... vs) {
  return v ? i : first(i + 1, vs...);
}

constexpr int last(int /*i*/, int result) { return result; }
template <typename T, typename... Ts>
constexpr int last(int i, int result, T v, Ts... vs) {
  return last(i + 1, v ? i : result, vs...);
}

/// Return the index of the first type in Ts which satisfies Predicate<T>.
/// Returns sizeof...(Ts) if none match.
template <template <typename> class Predicate, typename... Ts>
constexpr int constexpr_first() {
  return first(0, Predicate<Ts>::value...);
}

/// Return the index of the last type in Ts which satisfies Predicate<T>, or -1
/// if none match.
template <template <typename> class Predicate, typename... Ts>
constexpr int constexpr_last() {
  return last(0, -1, Predicate<Ts>::value...);
}

/// Return the Nth element from the parameter pack
template <size_t N, typename T, typename... Ts>
struct pack_element {
  using type = typename pack_element<N - 1, Ts...>::type;
};
template <typename T, typename... Ts>
struct pack_element<0, T, Ts...> {
  using type = T;
};

/// Return the one and only type which matches the predicate, or Default if none
/// match. If more than one type matches the predicate, fail at compile-time.
template <template <typename> class Predicate, typename Default, typename... Ts>
struct exactly_one {
  static constexpr auto found = constexpr_sum(Predicate<Ts>::value...);
  static_assert(found <= 1, "Found more than one type matching the predicate");

  static constexpr auto index = found ? constexpr_first<Predicate, Ts...>() : 0;
  using type =
      std::conditional_t<found, typename pack_element<index, Ts...>::type,
                         Default>;
};
template <template <typename> class P, typename Default>
struct exactly_one<P, Default> {
  using type = Default;
};

template <template <typename> class Predicate, typename Default, typename... Ts>
using exactly_one_t = typename exactly_one<Predicate, Default, Ts...>::type;

/// Apply a function over each element of a parameter pack
#ifdef __cpp_fold_expressions
#define TACHYON_EXPAND_SIDE_EFFECTS(PATTERN) (((PATTERN), void()), ...)
#else
using expand_side_effects = bool[];
#define TACHYON_EXPAND_SIDE_EFFECTS(PATTERN)       \
  (void)::tachyon::base::expand_side_effects { \
    ((PATTERN), void(), false)..., false         \
  }
#endif

// -------------------------------------------------------------------------------

template <typename T>
struct reference_to_pointer {
  using Type = T;
};

template <typename T>
struct reference_to_pointer<T&> {
  using Type = std::remove_reference_t<T>*;
};

template <typename T>
using reference_to_pointer_t = typename reference_to_pointer<T>::Type;

}  // namespace tachyon::base

#endif  // TACHYON_BASE_TEMPLATE_UTIL_H_

// clang-format on
