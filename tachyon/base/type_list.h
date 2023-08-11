// Copyright (c) 2011 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Derived from chromium/base/bind_internal.h

#ifndef TACHYON_BASE_TYPE_LIST_H_
#define TACHYON_BASE_TYPE_LIST_H_

#include <stddef.h>

#include <tuple>
#include <type_traits>

namespace tachyon::base::internal {

// Packs a list of types to hold them in a single type.
template <typename... Types>
struct TypeList {};

// Used for DropTypeListItem implementation.
template <size_t n, typename List>
struct DropTypeListItemImpl;

// Do not use enable_if and SFINAE here to avoid MSVC2013 compile failure.
template <size_t n, typename T, typename... List>
struct DropTypeListItemImpl<n, TypeList<T, List...>>
    : DropTypeListItemImpl<n - 1, TypeList<List...>> {};

template <typename T, typename... List>
struct DropTypeListItemImpl<0, TypeList<T, List...>> {
  using Type = TypeList<T, List...>;
};

template <>
struct DropTypeListItemImpl<0, TypeList<>> {
  using Type = TypeList<>;
};

// A type-level function that drops |n| list item from given TypeList.
template <size_t n, typename List>
using DropTypeListItem = typename DropTypeListItemImpl<n, List>::Type;

// Used for TakeTypeListItem implementation.
template <size_t n, typename List, typename... Accum>
struct TakeTypeListItemImpl;

// Do not use enable_if and SFINAE here to avoid MSVC2013 compile failure.
template <size_t n, typename T, typename... List, typename... Accum>
struct TakeTypeListItemImpl<n, TypeList<T, List...>, Accum...>
    : TakeTypeListItemImpl<n - 1, TypeList<List...>, Accum..., T> {};

template <typename T, typename... List, typename... Accum>
struct TakeTypeListItemImpl<0, TypeList<T, List...>, Accum...> {
  using Type = TypeList<Accum...>;
};

template <typename... Accum>
struct TakeTypeListItemImpl<0, TypeList<>, Accum...> {
  using Type = TypeList<Accum...>;
};

// A type-level function that takes first |n| list item from given TypeList.
// E.g. TakeTypeListItem<3, TypeList<A, B, C, D>> is evaluated to
// TypeList<A, B, C>.
template <size_t n, typename List>
using TakeTypeListItem = typename TakeTypeListItemImpl<n, List>::Type;

// Used for ConcatTypeLists implementation.
template <typename List1, typename List2>
struct ConcatTypeListsImpl;

template <typename... Types1, typename... Types2>
struct ConcatTypeListsImpl<TypeList<Types1...>, TypeList<Types2...>> {
  using Type = TypeList<Types1..., Types2...>;
};

// A type-level function that concats two TypeLists.
template <typename List1, typename List2>
using ConcatTypeLists = typename ConcatTypeListsImpl<List1, List2>::Type;

// Used for ConvertTypeListToTuple implementation.
template <typename... List>
struct ConvertTypeListToTupleImpl;

template <typename... List>
struct ConvertTypeListToTupleImpl<TypeList<List...>> {
  using Type = std::tuple<List...>;
};

// A type-level function that converts TypeList to std::tuple.
template <typename List>
using ConvertTypeListToTuple = typename ConvertTypeListToTupleImpl<List>::Type;

// Used for GetType implementation.
template <size_t idx, typename... List>
struct GetTypeImpl;

template <size_t idx, typename T, typename... List>
struct GetTypeImpl<idx, TypeList<T, List...>>
    : GetTypeImpl<idx - 1, TypeList<List...>> {};

template <typename T, typename... List>
struct GetTypeImpl<0, TypeList<T, List...>> {
  using Type = T;
};

// A type-level function that converts TypeList to std::tuple.
template <size_t idx, typename List>
using GetType = typename GetTypeImpl<idx, List>::Type;

template <typename... Types>
struct GetSizeImpl {};

template <typename... Types>
struct GetSizeImpl<TypeList<Types...>> {
  static constexpr size_t value = sizeof...(Types);
};

template <typename List>
inline constexpr size_t GetSize = GetSizeImpl<List>::value;

}  // namespace tachyon::base::internal

#endif  //  TACHYON_BASE_TYPE_LIST_H_
