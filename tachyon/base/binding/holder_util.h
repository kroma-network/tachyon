#ifndef TACHYON_BASE_BINDING_HOLDER_UTIL_H_
#define TACHYON_BASE_BINDING_HOLDER_UTIL_H_

#include <memory>

namespace tachyon::base {
namespace internal {

template <typename T, typename Holder>
struct is_holder_type : std::false_type {};

template <typename T>
struct is_holder_type<T, T> : std::true_type {};

template <typename T>
struct is_holder_type<T, std::shared_ptr<T>> : std::true_type {};

template <typename T, typename Deleter>
struct is_holder_type<T, std::unique_ptr<T, Deleter>> : std::true_type {};

template <typename Holder, typename Class>
struct HolderCreator;

// TODO(chokobole): Need to use universal reference and perfect forwarding
template <typename Class, typename... Args>
struct HolderCreator<Class, Class(Args...)> {
  static Class DoCreate(Args... args) {
    return Class(std::forward<Args>(args)...);
  }
};

// TODO(chokobole): Need to use universal reference and perfect forwarding
template <typename Class, typename... Args>
struct HolderCreator<std::shared_ptr<Class>, Class(Args...)> {
  static std::shared_ptr<Class> DoCreate(Args... args) {
    return std::shared_ptr<Class>(new Class(std::forward<Args>(args)...));
  }
};

// TODO(chokobole): Need to use universal reference and perfect forwarding
template <typename Deleter, typename Class, typename... Args>
struct HolderCreator<std::unique_ptr<Class, Deleter>, Class(Args...)> {
  static std::unique_ptr<Class, Deleter> DoCreate(Args... args) {
    return std::unique_ptr<Class, Deleter>(
        new Class(std::forward<Args>(args)...));
  }
};

template <typename T>
struct GetRawPtr {
  static T* Get(T& v) { return &v; }
};

template <typename T>
struct GetRawPtr<std::shared_ptr<T>> {
  static T* Get(std::shared_ptr<T>& v) { return v.get(); }
};

template <typename T, typename Deleter>
struct GetRawPtr<std::unique_ptr<T, Deleter>> {
  static T* Get(std::unique_ptr<T>& v) { return v.get(); }
};

}  // namespace internal
}  // namespace tachyon::base

#endif  // TACHYON_BASE_BINDING_HOLDER_UTIL_H_
