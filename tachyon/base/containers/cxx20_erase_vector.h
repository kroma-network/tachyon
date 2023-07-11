// Copyright 2021 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_BASE_CONTAINERS_CXX20_ERASE_VECTOR_H_
#define TACHYON_BASE_CONTAINERS_CXX20_ERASE_VECTOR_H_

#include <algorithm>
#include <iterator>
#include <vector>

namespace tachyon {
namespace base {

// Erase/EraseIf are based on C++20's uniform container erasure API:
// - https://eel.is/c++draft/libraryindex#:erase
// - https://eel.is/c++draft/libraryindex#:erase_if
// They provide a generic way to erase elements from a container.
// The functions here implement these for the standard containers until those
// functions are available in the C++ standard.
// Note: there is no std::erase for standard associative containers so we don't
// have it either.

template <class T, class Allocator, class Value>
size_t Erase(std::vector<T, Allocator>& container, const Value& value) {
  auto it = std::remove(container.begin(), container.end(), value);
  size_t removed = static_cast<size_t>(std::distance(it, container.end()));
  container.erase(it, container.end());
  return removed;
}

template <class T, class Allocator, class Predicate>
size_t EraseIf(std::vector<T, Allocator>& container, Predicate pred) {
  auto it = std::remove_if(container.begin(), container.end(), pred);
  size_t removed = static_cast<size_t>(std::distance(it, container.end()));
  container.erase(it, container.end());
  return removed;
}

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_BASE_CONTAINERS_CXX20_ERASE_VECTOR_H_
