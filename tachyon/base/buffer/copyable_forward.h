#ifndef TACHYON_BASE_BUFFER_COPYABLE_FORWARD_H_
#define TACHYON_BASE_BUFFER_COPYABLE_FORWARD_H_

#include <type_traits>

namespace tachyon::base {

class Buffer;

// NOTE: Do not implement for builtin serializable.
// See tachyon/base/buffer.h
template <typename T, typename SFINAE = void>
class Copyable;

template <typename, typename = void>
struct IsCopyable : std::false_type {};

template <typename T>
struct IsCopyable<
    T,
    std::void_t<decltype(Copyable<T>::WriteTo(std::declval<const T&>(),
                                              std::declval<Buffer*>())),
                decltype(Copyable<T>::ReadFrom(std::declval<const Buffer&>(),
                                               std::declval<T*>())),
                decltype(Copyable<T>::EstimateSize(std::declval<const T&>()))>>
    : std::true_type {};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_BUFFER_COPYABLE_FORWARD_H_
