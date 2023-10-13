#ifndef TACHYON_BASE_BINDING_CALLABLE_UTIL_H_
#define TACHYON_BASE_BINDING_CALLABLE_UTIL_H_

#include <type_traits>
#include <tuple>
#include <utility>

#include "tachyon/base/template_util.h"
#include "tachyon/base/type_list.h"

namespace tachyon::base {
namespace internal {

template <template <typename, typename SFINAE = void> class Predicate,
          typename T, typename SFINAE = void>
struct DeclarableTupleTypeImpl;

template <template <typename, typename SFINAE = void> class Predicate,
          typename T>
struct DeclarableTupleTypeImpl<
    Predicate, T, std::enable_if_t<Predicate<std::decay_t<T>>::value>> {
  using Type = std::decay_t<T>;
};

template <template <typename, typename SFINAE = void> class Predicate,
          typename T>
struct DeclarableTupleTypeImpl<
    Predicate, T, std::enable_if_t<!Predicate<std::decay_t<T>>::value>> {
  using Type = reference_to_pointer_t<T>;
};

template <template <typename, typename SFINAE = void> class Predicate,
          typename T>
using DeclarableTupleType =
    typename DeclarableTupleTypeImpl<Predicate, T>::Type;

template <template <typename, typename SFINAE = void> class Predicate,
          typename... List>
struct ConvertTypeListToDeclarableTupleImpl;

template <template <typename, typename SFINAE = void> class Predicate,
          typename... List>
struct ConvertTypeListToDeclarableTupleImpl<Predicate, TypeList<List...>> {
  using Type = std::tuple<DeclarableTupleType<Predicate, List>...>;
};

template <template <typename, typename SFINAE = void> class Predicate,
          typename List>
using ConvertTypeListToDeclarableTuple =
    typename ConvertTypeListToDeclarableTupleImpl<Predicate, List>::Type;

// pointer -> reference
template <typename R, typename T,
          std::enable_if_t<std::is_reference<R>::value &&
                           std::is_pointer<T>::value>* = nullptr>
R arg_cast(T v) {
  return *v;
}

// pointer -> pointer
template <typename R, typename T,
          std::enable_if_t<std::is_pointer<R>::value &&
                           std::is_pointer<T>::value>* = nullptr>
R arg_cast(T v) {
  return v;
}

// non-pointer -> non-pointer, movable
template <typename R, typename T,
          std::enable_if_t<std::is_move_constructible<R>::value &&
                           !std::is_pointer<T>::value>* = nullptr>
R arg_cast(T&& v) {
  return std::move(v);
}

// non-pointer -> non-pointer, not movable
template <typename R, typename T,
          std::enable_if_t<!std::is_move_constructible<R>::value &&
                           !std::is_pointer<T>::value>* = nullptr>
R arg_cast(T&& v) {
  return static_cast<R>(v);
}

template <template <typename, typename SFINAE = void> class Predicate>
struct RetTypeCaster {
  template <typename R, typename T,
            std::enable_if_t<Predicate<std::decay_t<R>>::value &&
                             std::is_move_constructible<R>::value>* = nullptr>
  static R cast(T&& v) {
    return std::move(v);
  }

  template <typename R, typename T,
            std::enable_if_t<Predicate<std::decay_t<R>>::value &&
                             !std::is_move_constructible<R>::value>* = nullptr>
  static R cast(T&& v) {
    return static_cast<R>(v);
  }

  template <typename R, typename T,
            std::enable_if_t<!Predicate<std::decay_t<R>>::value &&
                             std::is_reference<R>::value>* = nullptr>
  static auto cast(T&& v) {
    return std::ref(v);
  }

  template <typename R, typename T,
            std::enable_if_t<!Predicate<std::decay_t<R>>::value &&
                             !std::is_reference<R>::value>* = nullptr>
  static R cast(T&& v) {
    return std::move(v);
  }
};

}  // namespace internal
}  // namespace tachyon::base

#endif  // TACHYON_BASE_BINDING_CALLABLE_UTIL_H_
