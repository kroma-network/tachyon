// Copyright (c) 2011 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_BASE_TEMPLATE_UTIL_H_
#define TACHYON_BASE_TEMPLATE_UTIL_H_

#include <iterator>
#include <type_traits>

namespace tachyon {
namespace base {

// Implementation of C++20's std::remove_cvref.
//
// References:
// - https://en.cppreference.com/w/cpp/types/remove_cvref
// - https://wg21.link/meta.trans.other#lib:remove_cvref
template <typename T>
struct remove_cvref {
  using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

// Implementation of C++20's std::remove_cvref_t.
//
// References:
// - https://en.cppreference.com/w/cpp/types/remove_cvref
// - https://wg21.link/meta.type.synop#lib:remove_cvref_t
template <typename T>
using remove_cvref_t = typename remove_cvref<T>::type;

// Simplified implementation of C++20's std::iter_value_t.
// As opposed to std::iter_value_t, this implementation does not restrict
// the type of `Iter` and does not consider specializations of
// `indirectly_readable_traits`.
//
// Reference: https://wg21.link/readable.traits#2
template <typename Iter>
using iter_value_t =
    typename std::iterator_traits<remove_cvref_t<Iter>>::value_type;

// Simplified implementation of C++20's std::iter_reference_t.
// As opposed to std::iter_reference_t, this implementation does not restrict
// the type of `Iter`.
//
// Reference: https://wg21.link/iterator.synopsis#:~:text=iter_reference_t
template <typename Iter>
using iter_reference_t = decltype(*std::declval<Iter&>());

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_BASE_TEMPLATE_UTIL_H_
