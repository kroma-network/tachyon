// Copyright 2020 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/functional/identity.h"

#include <vector>

#include "gtest/gtest.h"

namespace tachyon {
namespace base {

TEST(FunctionalTest, Identity) {
  static constexpr identity id;

  std::vector<int> v;
  EXPECT_EQ(&v, &id(v));

  constexpr int arr = {0};
  static_assert(arr == id(arr), "");
}

}  // namespace base
}  // namespace tachyon
