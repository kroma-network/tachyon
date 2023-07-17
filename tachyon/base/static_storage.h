#ifndef TACHYON_BASE_STATIC_STORAGE_H_
#define TACHYON_BASE_STATIC_STORAGE_H_

#include <type_traits>

#define DEFINE_STATIC_STORAGE_TEMPLATE_METHOD(type, method_name)               \
  template <typename T = type,                                                 \
            std::enable_if_t<std::is_trivially_destructible_v<T>>* = nullptr>  \
  static T& method_name() {                                                    \
    static T storage;                                                          \
    return storage;                                                            \
  }                                                                            \
                                                                               \
  template <typename T = type,                                                 \
            std::enable_if_t<!std::is_trivially_destructible_v<T>>* = nullptr> \
  static T& method_name() {                                                    \
    static ::tachyon::base::NoDestructor<T> storage;                           \
    return *storage;                                                           \
  }

#endif  // TACHYON_BASE_STATIC_STORAGE_H_
