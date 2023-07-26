// Copyright 2017 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_BASE_NUMERICS_RANGES_H_
#define TACHYON_BASE_NUMERICS_RANGES_H_

#include <cmath>
#include <type_traits>

namespace tachyon {
namespace base {

template <typename T>
constexpr bool IsApproximatelyEqual(T lhs, T rhs, T tolerance) {
  static_assert(std::is_arithmetic<T>::value, "Argument must be arithmetic");
  return std::abs(rhs - lhs) <= tolerance;
}

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_BASE_NUMERICS_RANGES_H_
