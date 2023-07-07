// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_BASE_RANGES_ALGORITHM_H_
#define TACHYON_BASE_RANGES_ALGORITHM_H_

#include <type_traits>

namespace tachyon {
namespace base {
namespace internal {

// This alias is used below to restrict iterator based APIs to types for which
// `iterator_category` and the pre-increment and post-increment operators are
// defined. This is required in situations where otherwise an undesired overload
// would be chosen, e.g. copy_if. In spirit this is similar to C++20's
// std::input_or_output_iterator, a concept that each iterator should satisfy.
template <typename Iter, typename = decltype(++std::declval<Iter&>()),
          typename = decltype(std::declval<Iter&>()++)>
using iterator_category_t =
    typename std::iterator_traits<Iter>::iterator_category;

}  // namespace internal
}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_BASE_RANGES_ALGORITHM_H_
