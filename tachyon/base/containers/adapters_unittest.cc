// Copyright 2014 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/containers/adapters.h"

#include <vector>

#include "gtest/gtest.h"

namespace tachyon {
namespace base {

TEST(AdaptersTest, Reversed) {
  std::vector<int> v;
  v.push_back(3);
  v.push_back(2);
  v.push_back(1);
  int j = 0;
  for (int& i : Reversed(v)) {
    EXPECT_EQ(++j, i);
    i += 100;
  }
  EXPECT_EQ(103, v[0]);
  EXPECT_EQ(102, v[1]);
  EXPECT_EQ(101, v[2]);
}

TEST(AdaptersTest, ReversedArray) {
  int v[3] = {3, 2, 1};
  int j = 0;
  for (int& i : Reversed(v)) {
    EXPECT_EQ(++j, i);
    i += 100;
  }
  EXPECT_EQ(103, v[0]);
  EXPECT_EQ(102, v[1]);
  EXPECT_EQ(101, v[2]);
}

TEST(AdaptersTest, ReversedConst) {
  std::vector<int> v;
  v.push_back(3);
  v.push_back(2);
  v.push_back(1);
  const std::vector<int>& cv = v;
  int j = 0;
  for (int i : Reversed(cv)) {
    EXPECT_EQ(++j, i);
  }
}

}  // namespace base
}  // namespace tachyon
