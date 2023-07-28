// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_BASE_CXX20_IS_CONSTANT_EVALUATED_H_
#define TACHYON_BASE_CXX20_IS_CONSTANT_EVALUATED_H_

namespace tachyon::base {

// Implementation of C++20's std::is_constant_evaluated.
//
// References:
// - https://en.cppreference.com/w/cpp/types/is_constant_evaluated
// - https://wg21.link/meta.const.eval
constexpr bool is_constant_evaluated() noexcept {
  return __builtin_is_constant_evaluated();
}

}  // namespace tachyon::base

#endif  // TACHYON_BASE_CXX20_IS_CONSTANT_EVALUATED_H_
