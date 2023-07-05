// Copyright (c) 2011 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Derived from chromium/base/bind_internal.h

#ifndef TACHYON_BASE_FUNCTOR_TRAITS_H_
#define TACHYON_BASE_FUNCTOR_TRAITS_H_

#include <functional>

#include "absl/meta/type_traits.h"

#include "tachyon/base/type_list.h"

namespace tachyon {
namespace base {
namespace internal {

// Used for MakeFunctionType implementation.
template <typename R, typename ArgList>
struct MakeFunctionTypeImpl;

template <typename R, typename... Args>
struct MakeFunctionTypeImpl<R, TypeList<Args...>> {
  // MSVC 2013 doesn't support Type Alias of function types.
  // Revisit this after we update it to newer version.
  typedef R Type(Args...);
};

// A type-level function that constructs a function type that has |R| as its
// return type and has TypeLists items as its arguments.
template <typename R, typename ArgList>
using MakeFunctionType = typename MakeFunctionTypeImpl<R, ArgList>::Type;

// Used for ExtractArgs and ExtractReturnType.
template <typename Signature>
struct ExtractArgsImpl;

template <typename R, typename... Args>
struct ExtractArgsImpl<R(Args...)> {
  using ReturnType = R;
  using ArgsList = TypeList<Args...>;
};

// A type-level function that extracts function arguments into a TypeList.
// E.g. ExtractArgs<R(A, B, C)> is evaluated to TypeList<A, B, C>.
template <typename Signature>
using ExtractArgs = typename ExtractArgsImpl<Signature>::ArgsList;

// A type-level function that extracts the return type of a function.
// E.g. ExtractReturnType<R(A, B, C)> is evaluated to R.
template <typename Signature>
using ExtractReturnType = typename ExtractArgsImpl<Signature>::ReturnType;

template <typename Callable,
          typename Signature = decltype(&Callable::operator())>
struct ExtractCallableRunTypeImpl;

template <typename Callable, typename R, typename... Args>
struct ExtractCallableRunTypeImpl<Callable, R (Callable::*)(Args...)> {
  using Type = R(Args...);
};

template <typename Callable, typename R, typename... Args>
struct ExtractCallableRunTypeImpl<Callable, R (Callable::*)(Args...) const> {
  using Type = R(Args...);
};

// Evaluated to RunType of the given callable type.
// Example:
//   auto f = [](int, char*) { return 0.1; };
//   ExtractCallableRunType<decltype(f)>
//   is evaluated to
//   double(int, char*);
template <typename Callable>
using ExtractCallableRunType =
    typename ExtractCallableRunTypeImpl<Callable>::Type;

// IsCallableObject<Functor> is std::true_type if |Functor| has operator().
// Otherwise, it's std::false_type.
// Example:
//   IsCallableObject<void(*)()>::value is false.
//
//   struct Foo {};
//   IsCallableObject<void(Foo::*)()>::value is false.
//
//   int i = 0;
//   auto f = [i]() {};
//   IsCallableObject<decltype(f)>::value is false.
template <typename Functor, typename SFINAE = void>
struct IsCallableObject : std::false_type {};

template <typename Callable>
struct IsCallableObject<Callable, absl::void_t<decltype(&Callable::operator())>>
    : std::true_type {};

// FunctorTraits<>
//
// See description at top of file.
template <typename Functor, typename SFINAE = void>
struct FunctorTraits;

// For empty callable types.
// This specialization is intended to allow binding captureless lambdas, based
// on the fact that captureless lambdas are empty while capturing lambdas are
// not. This also allows any functors as far as it's an empty class.
// Example:
//
//   // Captureless lambdas are allowed.
//   []() {return 42;};
//
//   // Capturing lambdas are *not* allowed.
//   int x;
//   [x]() {return x;};
//
//   // Any empty class with operator() is allowed.
//   struct Foo {
//     void operator()() const {}
//     // No non-static member variable and no virtual functions.
//   };
template <typename Functor>
struct FunctorTraits<Functor,
                     std::enable_if_t<IsCallableObject<Functor>::value &&
                                      std::is_empty<Functor>::value>> {
  using RunType = ExtractCallableRunType<Functor>;
  using ReturnType = ExtractReturnType<RunType>;

  static constexpr bool is_method = false;
  static constexpr bool is_nullable = false;
  static constexpr bool is_callback = false;

  template <typename RunFunctor, typename... RunArgs>
  static ExtractReturnType<RunType> Invoke(RunFunctor&& functor,
                                           RunArgs&&... args) {
    return std::forward<RunFunctor>(functor)(std::forward<RunArgs>(args)...);
  }
};

// For functions.
template <typename R, typename... Args>
struct FunctorTraits<R (*)(Args...)> {
  using RunType = R(Args...);
  using ReturnType = R;

  static constexpr bool is_method = false;
  static constexpr bool is_nullable = true;
  static constexpr bool is_callback = false;

  template <typename Function, typename... RunArgs>
  static R Invoke(Function&& function, RunArgs&&... args) {
    return std::forward<Function>(function)(std::forward<RunArgs>(args)...);
  }
};

// For methods.
template <typename R, typename Receiver, typename... Args>
struct FunctorTraits<R (Receiver::*)(Args...)> {
  using RunType = R(Receiver*, Args...);
  using ReturnType = R;

  static constexpr bool is_method = true;
  static constexpr bool is_nullable = true;
  static constexpr bool is_callback = false;

  template <typename Method, typename ReceiverPtr, typename... RunArgs>
  static R Invoke(Method method, ReceiverPtr&& receiver_ptr,
                  RunArgs&&... args) {
    return ((*receiver_ptr).*method)(std::forward<RunArgs>(args)...);
  }
};

// For const methods.
template <typename R, typename Receiver, typename... Args>
struct FunctorTraits<R (Receiver::*)(Args...) const> {
  using RunType = R(const Receiver*, Args...);
  using ReturnType = R;

  static constexpr bool is_method = true;
  static constexpr bool is_nullable = true;
  static constexpr bool is_callback = false;

  template <typename Method, typename ReceiverPtr, typename... RunArgs>
  static R Invoke(Method method, ReceiverPtr&& receiver_ptr,
                  RunArgs&&... args) {
    return ((*receiver_ptr).*method)(std::forward<RunArgs>(args)...);
  }
};

#ifdef __cpp_noexcept_function_type
// noexcept makes a distinct function type in C++17.
// I.e. `void(*)()` and `void(*)() noexcept` are same in pre-C++17, and
// different in C++17.
template <typename R, typename... Args>
struct FunctorTraits<R (*)(Args...) noexcept> : FunctorTraits<R (*)(Args...)> {
};

template <typename R, typename Receiver, typename... Args>
struct FunctorTraits<R (Receiver::*)(Args...) noexcept>
    : FunctorTraits<R (Receiver::*)(Args...)> {};

template <typename R, typename Receiver, typename... Args>
struct FunctorTraits<R (Receiver::*)(Args...) const noexcept>
    : FunctorTraits<R (Receiver::*)(Args...) const> {};
#endif

// For lambda.
template <typename R, typename... Args>
struct FunctorTraits<std::function<R(Args...)>> {
  constexpr static bool is_method = false;
  using RunType = R(Args...);
  using ReturnType = ExtractReturnType<RunType>;
};

template <typename Functor>
using MakeFunctorTraits = FunctorTraits<std::decay_t<Functor>>;

// Extracts necessary type info from Functor and BoundArgs.
// Used to implement MakeUnboundRunType, BindOnce and BindRepeating.
template <typename Functor, typename... BoundArgs>
struct BindTypeHelper {
  static constexpr size_t num_bounds = sizeof...(BoundArgs);
  using FunctorTraits = MakeFunctorTraits<Functor>;

  // Example:
  //   When Functor is `double (Foo::*)(int, const std::string&)`, and BoundArgs
  //   is a template pack of `Foo*` and `int16_t`:
  //    - RunType is `double(Foo*, int, const std::string&)`,
  //    - ReturnType is `double`,
  //    - RunParamsList is `TypeList<Foo*, int, const std::string&>`,
  //    - BoundParamsList is `TypeList<Foo*, int>`,
  //    - UnboundParamsList is `TypeList<const std::string&>`,
  //    - BoundArgsList is `TypeList<Foo*, int16_t>`,
  //    - UnboundRunType is `double(const std::string&)`.
  using RunType = typename FunctorTraits::RunType;
  using ReturnType = ExtractReturnType<RunType>;

  using RunParamsList = ExtractArgs<RunType>;
  using BoundParamsList = TakeTypeListItem<num_bounds, RunParamsList>;
  using UnboundParamsList = DropTypeListItem<num_bounds, RunParamsList>;

  using BoundArgsList = TypeList<BoundArgs...>;

  using UnboundRunType = MakeFunctionType<ReturnType, UnboundParamsList>;
};

}  // namespace internal
}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_BASE_FUNCTOR_TRAITS_H_
