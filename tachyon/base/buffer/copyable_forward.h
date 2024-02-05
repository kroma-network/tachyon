#ifndef TACHYON_BASE_BUFFER_COPYABLE_FORWARD_H_
#define TACHYON_BASE_BUFFER_COPYABLE_FORWARD_H_

#include <type_traits>

#include "tachyon/base/types/cxx20_is_bounded_array.h"

namespace tachyon::base {

class Buffer;
class ReadOnlyBuffer;

// NOTE: Do not implement for builtin serializable.
// See tachyon/base/buffer/read_only_buffer.h
template <typename T, typename SFINAE = void>
class Copyable;

template <typename, typename = void>
struct IsCopyable : std::false_type {};

template <typename T>
struct IsCopyable<
    T,
    std::void_t<
        decltype(Copyable<T>::WriteTo(std::declval<const T&>(),
                                      std::declval<Buffer*>())),
        decltype(Copyable<T>::ReadFrom(
            std::declval<const ReadOnlyBuffer&>(),
            std::declval<std::conditional_t<is_bounded_array_v<T>, T, T*>>())),
        decltype(Copyable<T>::EstimateSize(std::declval<const T&>()))>>
    : std::true_type {};

template <typename T>
size_t EstimateSize(const T& value) {
  return Copyable<T>::EstimateSize(value);
}

}  // namespace tachyon::base

#endif  // TACHYON_BASE_BUFFER_COPYABLE_FORWARD_H_
