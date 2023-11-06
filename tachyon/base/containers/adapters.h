// Copyright 2014 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_BASE_CONTAINERS_ADAPTERS_H_
#define TACHYON_BASE_CONTAINERS_ADAPTERS_H_

#include <stddef.h>

#include <algorithm>
#include <iterator>
#include <tuple>
#include <utility>

#include "absl/types/span.h"

#include "tachyon/base/numerics/checked_math.h"
#include "tachyon/base/template_util.h"

namespace tachyon::base {
namespace internal {

// Internal adapter class for implementing tachyon::base::Reversed.
template <typename T>
class ReversedAdapter {
 public:
  using Iterator = decltype(std::rbegin(std::declval<T&>()));

  explicit ReversedAdapter(T& t) : t_(t) {}
  ReversedAdapter(const ReversedAdapter& ra) : t_(ra.t_) {}

  ReversedAdapter& operator=(const ReversedAdapter& ra) = delete;

  Iterator begin() const { return std::rbegin(t_); }
  Iterator end() const { return std::rend(t_); }

 private:
  T& t_;
};

template <typename T>
class ChunkedAdapter {
 public:
  class Iterator {
   public:
    using ItType = decltype(std::begin(std::declval<T&>()));

    using underlying_value_type =
        std::conditional_t<std::is_const_v<T>, const iter_value_t<ItType>,
                           iter_value_t<ItType>>;

    using difference_type = std::ptrdiff_t;
    using value_type = absl::Span<underlying_value_type>;
    using pointer = value_type*;
    using reference = value_type&;
    using iterator_category = std::forward_iterator_tag;

    static Iterator begin(T& t, size_t chunk_size) {
      return Iterator(std::begin(t), chunk_size,
                      std::distance(std::begin(t), std::end(t)));
    }
    static Iterator end(T& t) { return Iterator(std::end(t)); }

    Iterator(const Iterator& other) = default;
    Iterator& operator=(const Iterator& other) = default;

    bool operator==(const Iterator& other) const { return it_ == other.it_; }
    bool operator!=(const Iterator& other) const { return !(*this == other); }

    Iterator& operator++() {
      it_ += len_;
      offset_ += len_;
      base::CheckedNumeric<size_t> len = offset_;
      len += chunk_size_;
      size_t new_len = len.ValueOrDie();
      if (new_len > size_) {
        len_ = size_ - offset_;
      } else {
        len_ = chunk_size_;
      }
      value_ = value_type(&*it_, len_);
      return *this;
    }

    Iterator operator++(int) {
      Iterator it(*this);
      ++(*this);
      return it;
    }

    // NOTE(chokobole): To suppress -Werror,-Wignored-reference-qualifiers on
    // mac
    const std::remove_const_t<pointer> operator->() const { return &value_; }
    pointer operator->() { return &value_; }

    // NOTE(chokobole): To suppress -Werror,-Wignored-reference-qualifiers on
    // mac
    const std::remove_const_t<reference> operator*() const { return value_; }
    reference operator*() { return value_; }

   private:
    explicit Iterator(ItType it) : it_(it), chunk_size_(0), size_(0), len_(0) {}
    Iterator(ItType it, size_t chunk_size, size_t size)
        : it_(it),
          chunk_size_(chunk_size),
          size_(size),
          len_(std::min(chunk_size, size)),
          value_(value_type(&*it, len_)) {}

    ItType it_;
    size_t offset_ = 0;
    size_t chunk_size_;
    size_t size_;
    size_t len_;
    value_type value_;
  };

  ChunkedAdapter(T& t, size_t chunk_size) : t_(t), chunk_size_(chunk_size) {}
  ChunkedAdapter(const ChunkedAdapter& ca) : t_(ca.t_) {}

  ChunkedAdapter& operator=(const ChunkedAdapter& ca) = delete;

  Iterator begin() const { return Iterator::begin(t_, chunk_size_); }
  Iterator end() const { return Iterator::end(t_); }

 private:
  T& t_;
  size_t chunk_size_;
};

template <typename T, typename U>
class ZippedAdapter {
 public:
  class Iterator {
   public:
    using ItType = decltype(std::begin(std::declval<T&>()));
    using ItType2 = decltype(std::begin(std::declval<U&>()));

    using underlying_value_type = iter_value_t<ItType>;
    using underlying_value_type2 = iter_value_t<ItType2>;

    using difference_type = std::ptrdiff_t;
    using value_type =
        std::tuple<underlying_value_type, underlying_value_type2>;
    using pointer = value_type*;
    using reference = value_type&;
    using iterator_category = std::forward_iterator_tag;

    static Iterator begin(T& t, U& u) {
      return Iterator(std::begin(t), std::begin(u));
    }
    static Iterator end(T& t, U& u) {
      return Iterator(std::end(t), std::end(u));
    }

    Iterator(const Iterator& other) = default;
    Iterator& operator=(const Iterator& other) = default;

    bool operator==(const Iterator& other) const {
      return it_ == other.it_ && it2_ == other.it2_;
    }
    bool operator!=(const Iterator& other) const { return !(*this == other); }

    Iterator& operator++() {
      ++it_;
      ++it2_;
      value_ = value_type(*it_, *it2_);
      return *this;
    }

    Iterator operator++(int) {
      Iterator it(*this);
      ++(*this);
      return it;
    }

    // NOTE(chokobole): To suppress -Werror,-Wignored-reference-qualifiers on
    // mac
    const std::remove_const_t<pointer> operator->() const { return &value_; }
    pointer operator->() { return &value_; }

    // NOTE(chokobole): To suppress -Werror,-Wignored-reference-qualifiers on
    // mac
    const std::remove_const_t<reference> operator*() const { return value_; }
    reference operator*() { return value_; }

   private:
    Iterator(ItType it, ItType2 it2)
        : it_(it), it2_(it2), value_(value_type(*it, *it2)) {}

    ItType it_;
    ItType2 it2_;
    value_type value_;
  };

  ZippedAdapter(T& t, U& u) : t_(t), u_(u) {}
  ZippedAdapter(const ZippedAdapter& za) : t_(za.t_), u_(za.u_) {}

  ZippedAdapter& operator=(const ZippedAdapter& za) = delete;

  Iterator begin() const { return Iterator::begin(t_, u_); }
  Iterator end() const { return Iterator::end(t_, u_); }

 private:
  T& t_;
  U& u_;
};

// Internal adapter class for implementing tachyon::base::Chained.
template <typename T>
class ChainedAdapter {
 public:
  class Iterator {
   public:
    using ItType = decltype(std::begin(std::declval<T&>()));

    using difference_type = std::ptrdiff_t;
    using value_type =
        std::conditional_t<std::is_const_v<T>, const iter_value_t<ItType>,
                           iter_value_t<ItType>>;
    using pointer = value_type*;
    using reference = value_type&;
    using iterator_category = std::forward_iterator_tag;

    static Iterator begin(T& first, T& second) {
      return Iterator(std::begin(first), std::end(first), std::begin(second),
                      std::end(second), true);
    }
    static Iterator end(T& first, T& second) {
      return Iterator(std::end(second), std::end(first), std::begin(second),
                      std::end(second), false);
    }

    Iterator(const Iterator& other) = default;
    Iterator& operator=(const Iterator& other) = default;

    bool operator==(const Iterator& other) const { return it_ == other.it_; }
    bool operator!=(const Iterator& other) const { return !(*this == other); }

    Iterator& operator++() {
      if (is_first_) {
        it_++;
        if (it_ == end_first_) {
          it_ = begin_second_;
          is_first_ = false;
        }
      } else {
        if (it_ != end_second_) {
          ++it_;
        }
      }
      return *this;
    }

    Iterator operator++(int) {
      Iterator it(*this);
      ++(*this);
      return it;
    }

    pointer operator->() { return &*it_; }

    reference operator*() { return *it_; }

   private:
    Iterator(ItType it, ItType end_first, ItType begin_second,
             ItType end_second, bool is_first)
        : it_(it),
          end_first_(end_first),
          begin_second_(begin_second),
          end_second_(end_second),
          is_first_(is_first) {}

    ItType it_;
    const ItType end_first_;
    const ItType begin_second_;
    const ItType end_second_;
    bool is_first_;
  };

  ChainedAdapter(T& first, T& second) : first_(first), second_(second) {}
  ChainedAdapter(const ChainedAdapter& ca)
      : first_(ca.first_), second_(ca.second_) {}

  ChainedAdapter& operator=(const ChainedAdapter& ca) = delete;

  Iterator begin() const { return Iterator::begin(first_, second_); }
  Iterator end() const { return Iterator::end(first_, second_); }

 private:
  T& first_;
  T& second_;
};

}  // namespace internal

// Reversed returns a container adapter usable in a range-based "for" statement
// for iterating a reversible container in reverse order.
//
// Example:
//
//   std::vector<int> v = {1, 2, 3, 4, 5};
//   for (int i : tachyon::base::Reversed(v)) {
//     // iterates through v from back to front
//     // 5, 4, 3, 2, 1
//   }
template <typename T>
internal::ReversedAdapter<T> Reversed(T& t) {
  return internal::ReversedAdapter<T>(t);
}

// Chunked returns a container adapter that can be used in a range-based "for"
// loop to iterate over a container in chunks of a specified size.
//
// Example:
//
//   std::vector<int> v = {1, 2, 3, 4, 5};
//   for (const absl::Span<int>& i : tachyon::base::Chunked(v, 2)) {
//     // iterates through v in chunks
//     // {1, 2}, {3, 4}, {5}
//   }
template <typename T>
internal::ChunkedAdapter<T> Chunked(T& t, size_t chunk_size) {
  return internal::ChunkedAdapter<T>(t, chunk_size);
}

// Zipped returns a container adapter that can be used in a range-based "for"
// loop to iterate over 2 containers.
// Note that in order to iterate safely, both lengths of the containers should
// be same.
//
// Example:
//
//   std::vector<int> v = {1, 2, 3};
//   std::vector<int> w = {4, 5, 6};
//   for (const std::tuple<int, int>& i : tachyon::base::Zipped(v, w)) {
//     // iterates through v in tuple
//     // {1, 4}, {2, 5}, {3, 6}
//   }
template <typename T, typename U>
internal::ZippedAdapter<T, U> Zipped(T& t, U& u) {
  return internal::ZippedAdapter<T, U>(t, u);
}

// Chained returns a container adapter that can be used in a range-based "for"
// loop to iterate over two containers of the same type as if they were a single
// container.
//
// Example:
//
//   std::vector<int> v = {1, 2, 3};
//   std::vector<int> w = {4, 5, 6};
//   for (int i : tachyon::base::Chained(v, w)) {
//     // iterates through v and w consecutively
//     // 1, 2, 3, 4, 5, 6
//   }
template <typename T>
internal::ChainedAdapter<T> Chained(T& first, T& second) {
  return internal::ChainedAdapter<T>(first, second);
}

}  // namespace tachyon::base

#endif  // TACHYON_BASE_CONTAINERS_ADAPTERS_H_
