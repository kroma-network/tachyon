#ifndef TACHYON_NODE_BASE_NODE_INTERNALS_FORWARDS_H_
#define TACHYON_NODE_BASE_NODE_INTERNALS_FORWARDS_H_

#if defined(TACHYON_NODE_BINDING)

#include "absl/meta/type_traits.h"
#include "third_party/node_addon_api/napi.h"

namespace tachyon::node {
namespace internal {

template <typename T, typename SFINAE = void>
class CppValueTraits;

template <typename, typename SFINAE = void>
struct JSCppTypeWrapper;

template <typename T>
using JSCppType = typename JSCppTypeWrapper<T>::Type;

template <typename, typename SFINAE = void>
struct NativeCppTypeWrapper;

template <typename T>
using NativeCppType = typename NativeCppTypeWrapper<T>::Type;

template <typename T>
bool ToNativeValue(const Napi::Value& value, T* v);

template <typename T>
Napi::Value ToJSValue(const Napi::CallbackInfo& info, T&& value);

template <typename, typename = void>
struct HasToNativeValue : std::false_type {};

template <typename T>
struct HasToNativeValue<T,
                        absl::void_t<decltype(CppValueTraits<T>::ToNativeValue(
                            std::declval<Napi::Value>(), std::declval<T*>()))>>
    : std::true_type {};

template <typename, typename = void>
struct HasToJSValue : std::false_type {};

template <typename T>
struct HasToJSValue<
    T, absl::void_t<decltype(CppValueTraits<T>::ToJSValue(
           std::declval<Napi::CallbackInfo>(), std::declval<T>()))>>
    : std::true_type {};

}  // namespace internal
}  // namespace tachyon::node

#endif  // defined(TACHYON_NODE_BINDING)

#endif  // TACHYON_NODE_BASE_NODE_INTERNALS_FORWARDS_H_
