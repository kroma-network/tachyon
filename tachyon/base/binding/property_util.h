#ifndef TACHYON_BASE_BINDING_PROPERTY_UTIL_H_
#define TACHYON_BASE_BINDING_PROPERTY_UTIL_H_

#include <functional>
#include <type_traits>

namespace tachyon::base {
namespace internal {

template <template <typename, typename SFINAE = void> class Predicate>
struct PropertyTypeCaster {
  template <typename T,
            std::enable_if_t<Predicate<std::decay_t<T>>::value>* = nullptr>
  static auto cast(T&& v) {
    return std::forward<T>(v);
  }

  template <typename T,
            std::enable_if_t<!Predicate<std::decay_t<T>>::value>* = nullptr>
  static auto cast(T&& v) {
    return std::ref(v);
  }
};

}  // namespace internal
}  // namespace tachyon::base

#endif  // TACHYON_BASE_BINDING_PROPERTY_UTIL_H_
