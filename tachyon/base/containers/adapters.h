// Copyright 2014 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_BASE_CONTAINERS_ADAPTERS_H_
#define TACHYON_BASE_CONTAINERS_ADAPTERS_H_

#include "tachyon/base/containers/chunked_iterator.h"
#include "tachyon/base/containers/zipped_iterator.h"

namespace tachyon::base {
namespace internal {

// Internal adapter class for implementing tachyon::base::Reversed.
template <typename T>
class ReversedAdapter {
 public:
  explicit ReversedAdapter(T& t) : t_(t) {}
  ReversedAdapter(const ReversedAdapter& ra) : t_(ra.t_) {}

  ReversedAdapter& operator=(const ReversedAdapter& ra) = delete;

  auto begin() const {
    if constexpr (std::is_const_v<T>) {
      return std::crbegin(t_);
    } else {
      return std::rbegin(t_);
    }
  }
  auto end() const {
    if constexpr (std::is_const_v<T>) {
      return std::crend(t_);
    } else {
      return std::rend(t_);
    }
  }

 private:
  T& t_;
};

template <typename T>
class ChunkedAdapter {
 public:
  ChunkedAdapter(T& t, size_t chunk_size) : t_(t), chunk_size_(chunk_size) {}
  ChunkedAdapter(const ChunkedAdapter& ca) : t_(ca.t_) {}

  ChunkedAdapter& operator=(const ChunkedAdapter& ca) = delete;

  auto begin() const {
    if constexpr (std::is_const_v<T>) {
      return ChunkedConstBegin(t_, chunk_size_);
    } else {
      return ChunkedBegin(t_, chunk_size_);
    }
  }
  auto end() const {
    if constexpr (std::is_const_v<T>) {
      return ChunkedConstEnd(t_, chunk_size_);
    } else {
      return ChunkedEnd(t_, chunk_size_);
    }
  }

 private:
  T& t_;
  size_t chunk_size_;
};

template <typename T, typename U>
class ZippedAdapter {
 public:
  ZippedAdapter(T& t, U& u) : t_(t), u_(u) {}
  ZippedAdapter(const ZippedAdapter& za) : t_(za.t_), u_(za.u_) {}

  ZippedAdapter& operator=(const ZippedAdapter& za) = delete;

  auto begin() const { return ZippedBegin(t_, u_); }
  auto end() const { return ZippedEnd(t_, u_); }

 private:
  T& t_;
  U& u_;
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

}  // namespace tachyon::base

#endif  // TACHYON_BASE_CONTAINERS_ADAPTERS_H_
