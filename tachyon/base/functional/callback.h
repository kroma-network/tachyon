// Copyright 2012 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_BASE_FUNCTIONAL_CALLBACK_H_
#define TACHYON_BASE_FUNCTIONAL_CALLBACK_H_

#include <functional>
#include <utility>
#include <type_traits>

#include "absl/functional/function_ref.h"

#include "tachyon/base/functional/callback_forward.h"

namespace tachyon::base {

template <typename R, typename... Args>
class OnceCallback<R(Args...)> {
 public:
  typedef std::function<R(Args...)> CallbackTy;

  constexpr OnceCallback() = default;
  OnceCallback(std::nullptr_t) = delete;
  OnceCallback(absl::FunctionRef<R(Args...)> callback) : callback_(callback) {}
  template <typename T,
            std::enable_if_t<std::is_convertible_v<T, CallbackTy>>* = nullptr>
  OnceCallback(T&& callback) : callback_(callback) {}
  OnceCallback(const OnceCallback& other) = delete;
  OnceCallback& operator=(const OnceCallback& other) = delete;
  OnceCallback(OnceCallback&& other) noexcept = default;
  OnceCallback& operator=(OnceCallback&& other) noexcept = default;

  OnceCallback(const RepeatingCallback<R(Args...)>& other) noexcept
      : callback_(other.callback_) {}
  OnceCallback& operator=(const RepeatingCallback<R(Args...)>& other) noexcept {
    callback_ = other.callback_;
    return *this;
  }

  R Run(Args... args) && {
    CallbackTy callback = callback_;
    callback_ = nullptr;
    return callback(std::forward<Args>(args)...);
  }

  operator bool() const { return static_cast<bool>(callback_); }

  bool is_null() const { return !static_cast<bool>(callback_); }

  void Reset() { callback_ = nullptr; }

 private:
  CallbackTy callback_;
};

template <typename R, typename... Args>
class RepeatingCallback<R(Args...)> {
 public:
  typedef std::function<R(Args...)> CallbackTy;

  constexpr RepeatingCallback() = default;
  RepeatingCallback(std::nullptr_t) = delete;
  RepeatingCallback(absl::FunctionRef<R(Args...)> callback)
      : callback_(callback) {}
  template <typename T,
            std::enable_if_t<std::is_convertible_v<T, CallbackTy>>* = nullptr>
  RepeatingCallback(T&& callback) : callback_(callback) {}
  RepeatingCallback(const RepeatingCallback& other) = default;
  RepeatingCallback& operator=(const RepeatingCallback& other) = default;

  R Run(Args... args) const& { return callback_(std::forward<Args>(args)...); }

  R Run(Args... args) && {
    CallbackTy callback = callback_;
    callback_ = nullptr;
    return callback(std::forward<Args>(args)...);
  }

  operator bool() const { return static_cast<bool>(callback_); }

  bool is_null() const { return !static_cast<bool>(callback_); }

  void Reset() { callback_ = nullptr; }

 private:
  friend class OnceCallback<R(Args...)>;
  CallbackTy callback_;
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_FUNCTIONAL_CALLBACK_H_
