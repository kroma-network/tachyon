// Copyright 2018 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// clang-format off

#ifndef TACHYON_BASE_CONTAINERS_UTIL_H_
#define TACHYON_BASE_CONTAINERS_UTIL_H_

#include <stdint.h>

namespace tachyon::base {

// TODO(crbug.com/817982): What we really need is for checked_math.h to be
// able to do checked arithmetic on pointers.
template <typename T>
inline uintptr_t get_uintptr(const T* t) {
  return reinterpret_cast<uintptr_t>(t);
}

}  // namespace tachyon::base

#endif  // TACHYON_BASE_CONTAINERS_UTIL_H_

// clang-format on
