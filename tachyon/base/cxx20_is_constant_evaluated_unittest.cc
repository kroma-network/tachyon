// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/cxx20_is_constant_evaluated.h"

#include "gtest/gtest.h"

namespace tachyon::base {

TEST(Cxx20IsConstantEvaluated, Basic) {
  static_assert(is_constant_evaluated(), "");
  EXPECT_FALSE(is_constant_evaluated());
}

}  // namespace tachyon::base
