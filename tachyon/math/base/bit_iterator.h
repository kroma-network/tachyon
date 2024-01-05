// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_BASE_BIT_ITERATOR_H_
#define TACHYON_MATH_BASE_BIT_ITERATOR_H_

#include <iterator>
#include <limits>

#include "tachyon/build/build_config.h"
#include "tachyon/math/base/bit_traits_forward.h"

namespace tachyon::math {

template <typename T>
class BitIteratorBE {
 public:
  using difference_type = std::ptrdiff_t;
  using value_type = bool;
  using pointer = bool;
  using reference = bool;
  using iterator_category = std::forward_iterator_tag;

  constexpr explicit BitIteratorBE(const T* value) : BitIteratorBE(value, 0) {}
  constexpr BitIteratorBE(const T* value, size_t index)
      : value_(value), index_(index) {}
  constexpr BitIteratorBE(const BitIteratorBE& other) = default;
  constexpr BitIteratorBE& operator=(const BitIteratorBE& other) = default;

  constexpr static BitIteratorBE begin(const T* value,
                                       bool skip_leading_zeros = false) {
#if ARCH_CPU_BIG_ENDIAN
    BitIteratorBE ret(value, 0);
#else  // ARCH_CPU_LITTLE_ENDIAN
    size_t bits = BitTraits<T>::GetNumBits(*value);
    if constexpr (BitTraits<T>::kIsDynamic) {
      if (bits == 0) {
        return end(value);
      }
    }
    BitIteratorBE ret(value, bits - 1);
#endif
    if (skip_leading_zeros) {
      while (!(*ret)) {
        ++ret;
#if ARCH_CPU_BIG_ENDIAN
        if (ret.index_ == bits) break;
#else  // ARCH_CPU_LITTLE_ENDIAN
        if (ret.index_ == std::numeric_limits<size_t>::max()) break;
#endif
      }
    }
    return ret;
  }

  constexpr static BitIteratorBE end(const T* value) {
#if ARCH_CPU_BIG_ENDIAN
    size_t bits = BitTraits<T>::GetNumBits(*value);
    if (bits == 0) {
      return begin(value);
    }
    return BitIteratorBE(value, bits);
#else  // ARCH_CPU_LITTLE_ENDIAN
    return BitIteratorBE(value, std::numeric_limits<size_t>::max());
#endif
  }

  constexpr bool operator==(const BitIteratorBE& other) const {
    return value_ == other.value_ && index_ == other.index_;
  }
  constexpr bool operator!=(const BitIteratorBE& other) const {
    return !(*this == other);
  }

  constexpr BitIteratorBE& operator++() {
#if ARCH_CPU_BIG_ENDIAN
    ++index_;
#else  // ARCH_CPU_LITTLE_ENDIAN
    --index_;
#endif
    return *this;
  }

  constexpr BitIteratorBE operator++(int) {
    BitIteratorBE it(*this);
    ++(*this);
    return it;
  }

  constexpr bool operator->() const {
    return BitTraits<T>::TestBit(*value_, index_);
  }

  constexpr bool operator*() const {
    return BitTraits<T>::TestBit(*value_, index_);
  }

 private:
  const T* value_ = nullptr;
  size_t index_ = 0;
};

template <typename T>
class BitIteratorLE {
 public:
  using difference_type = std::ptrdiff_t;
  using value_type = bool;
  using pointer = bool;
  using reference = bool;
  using iterator_category = std::forward_iterator_tag;

  constexpr explicit BitIteratorLE(const T* value) : BitIteratorLE(value, 0) {}
  constexpr BitIteratorLE(const T* value, size_t index)
      : value_(value), index_(index) {}
  constexpr BitIteratorLE(const BitIteratorLE& other) = default;
  constexpr BitIteratorLE& operator=(const BitIteratorLE& other) = default;

  constexpr static BitIteratorLE begin(const T* value) {
#if ARCH_CPU_LITTLE_ENDIAN
    return BitIteratorLE(value, 0);
#else  // ARCH_CPU_BIG_ENDIAN
    size_t bits = BitTraits<T>::GetNumBits(*value);
    if constexpr (BitTraits<T>::kIsDynamic) {
      if (bits == 0) {
        return end(value);
      }
    }
    return BitIteratorLE(value, bits - 1);
#endif
  }

  constexpr static BitIteratorLE end(const T* value,
                                     bool skip_trailing_zeros = false) {
#if ARCH_CPU_LITTLE_ENDIAN
    size_t bits = BitTraits<T>::GetNumBits(*value);
    if (bits == 0) {
      return begin(value);
    }
    BitIteratorLE ret(value, bits);
#else  // ARCH_CPU_BIG_ENDIAN
    BitIteratorLE ret(value, std::numeric_limits<size_t>::max());
#endif
    if (!skip_trailing_zeros) return ret;
    while (!(*ret)) {
      --ret;
#if ARCH_CPU_LITTLE_ENDIAN
      if (ret.index_ == std::numeric_limits<size_t>::max()) break;
#else  // ARCH_CPU_BIG_ENDIAN
      if (ret.index_ == bits) break;
#endif
    }
    return ++ret;
  }

  bool operator==(const BitIteratorLE& other) const {
    return value_ == other.value_ && index_ == other.index_;
  }
  bool operator!=(const BitIteratorLE& other) const {
    return !(*this == other);
  }

  constexpr BitIteratorLE& operator++() {
#if ARCH_CPU_LITTLE_ENDIAN
    ++index_;
#else  // ARCH_CPU_BIG_ENDIAN
    --index_;
#endif
    return *this;
  }

  constexpr BitIteratorLE operator++(int) {
    BitIteratorLE it(*this);
    ++(*this);
    return it;
  }

  constexpr bool operator->() const {
    return BitTraits<T>::TestBit(*value_, index_);
  }

  constexpr bool operator*() const {
    return BitTraits<T>::TestBit(*value_, index_);
  }

 private:
  constexpr BitIteratorLE& operator--() {
#if ARCH_CPU_LITTLE_ENDIAN
    --index_;
#else  // ARCH_CPU_BIG_ENDIAN
    ++index_;
#endif
    return *this;
  }

  constexpr BitIteratorLE operator--(int) {
    BitIteratorLE it(*this);
    --(*this);
    return it;
  }

  const T* value_ = nullptr;
  size_t index_ = 0;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_BASE_BIT_ITERATOR_H_
