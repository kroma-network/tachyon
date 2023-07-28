// Copyright 2011 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_BASE_CALLBACK_FORWARD_H_
#define TACHYON_BASE_CALLBACK_FORWARD_H_

namespace tachyon::base {

template <typename Signature>
class OnceCallback;

template <typename Signature>
class RepeatingCallback;

// Syntactic sugar to make OnceClosure<void()> and RepeatingClosure<void()>
// easier to declare since they will be used in a lot of APIs with delayed
// execution.
using OnceClosure = OnceCallback<void()>;
using RepeatingClosure = RepeatingCallback<void()>;

}  // namespace tachyon::base

#endif  // TACHYON_BASE_CALLBACK_FORWARD_H_
