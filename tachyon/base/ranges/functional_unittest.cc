// Copyright 2020 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/ranges/functional.h"

#include "gtest/gtest.h"

namespace tachyon {
namespace base {

TEST(RangesTest, EqualTo) {
  ranges::equal_to eq;
  EXPECT_TRUE(eq(0, 0));
  EXPECT_FALSE(eq(0, 1));
  EXPECT_FALSE(eq(1, 0));
}

TEST(RangesTest, Less) {
  ranges::less lt;
  EXPECT_FALSE(lt(0, 0));
  EXPECT_TRUE(lt(0, 1));
  EXPECT_FALSE(lt(1, 0));
}

}  // namespace base
}  // namespace tachyon
