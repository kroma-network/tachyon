// Copyright 2020 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_BASE_FUNCTIONAL_IDENTITY_H_
#define TACHYON_BASE_FUNCTIONAL_IDENTITY_H_

#include <utility>

namespace tachyon::base {

// Implementation of C++20's std::identity.
//
// Reference:
// - https://en.cppreference.com/w/cpp/utility/functional/identity
// - https://wg21.link/func.identity
struct identity {
  template <typename T>
  constexpr T&& operator()(T&& t) const noexcept {
    return std::forward<T>(t);
  }

  using is_transparent = void;
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_FUNCTIONAL_IDENTITY_H_
