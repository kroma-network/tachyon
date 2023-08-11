// Copyright 2021 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_BASE_CONTAINERS_CXX20_ERASE_INTERNAL_H_
#define TACHYON_BASE_CONTAINERS_CXX20_ERASE_INTERNAL_H_

#include <stddef.h>

// Internal portion of base/containers/cxx20_erase_*.h. Please include those
// headers instead of including this directly.

namespace tachyon::base {

namespace internal {

// Calls erase on iterators of matching elements and returns the number of
// removed elements.
template <typename Container, typename Predicate>
size_t IterateAndEraseIf(Container& container, Predicate pred) {
  size_t old_size = container.size();
  for (auto it = container.begin(), last = container.end(); it != last;) {
    if (pred(*it))
      it = container.erase(it);
    else
      ++it;
  }
  return old_size - container.size();
}

}  // namespace internal

}  // namespace tachyon::base

#endif  // TACHYON_BASE_CONTAINERS_CXX20_ERASE_INTERNAL_H_
