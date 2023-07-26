// Copyright 2020 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/containers/contains.h"

#include <set>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/ascii.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/base/functional/identity.h"

namespace tachyon {
namespace base {

TEST(ContainsTest, GenericContains) {
  constexpr char allowed_chars[] = {'a', 'b', 'c', 'd'};

  static_assert(Contains(allowed_chars, 'a'), "");
  static_assert(!Contains(allowed_chars, 'z'), "");
  static_assert(!Contains(allowed_chars, 0), "");

  constexpr char allowed_chars_including_nul[] = "abcd";
  static_assert(Contains(allowed_chars_including_nul, 0), "");
}

TEST(ContainsTest, GenericContainsWithProjection) {
  const char allowed_chars[] = {'A', 'B', 'C', 'D'};

  EXPECT_TRUE(Contains(allowed_chars, 'a', &absl::ascii_tolower));
  EXPECT_FALSE(Contains(allowed_chars, 'z', &absl::ascii_tolower));
  EXPECT_FALSE(Contains(allowed_chars, 0, &absl::ascii_tolower));
}

TEST(ContainsTest, GenericSetContainsWithProjection) {
  constexpr std::string_view kFoo = "foo";
  std::set<std::string> set = {"foo", "bar", "baz"};

  // Opt into a linear search by explicitly providing a projection:
  EXPECT_TRUE(Contains(set, kFoo, identity{}));
}

TEST(ContainsTest, ContainsWithFindAndNpos) {
  std::string str = "abcd";

  EXPECT_TRUE(Contains(str, 'a'));
  EXPECT_FALSE(Contains(str, 'z'));
  EXPECT_FALSE(Contains(str, 0));
}

TEST(ContainsTest, ContainsWithFindAndEnd) {
  std::set<int> set = {1, 2, 3, 4};

  EXPECT_TRUE(Contains(set, 1));
  EXPECT_FALSE(Contains(set, 5));
  EXPECT_FALSE(Contains(set, 0));
}

TEST(ContainsTest, ContainsWithContains) {
  absl::flat_hash_set<int> set = {1, 2, 3, 4};

  EXPECT_TRUE(Contains(set, 1));
  EXPECT_FALSE(Contains(set, 5));
  EXPECT_FALSE(Contains(set, 0));
}

}  // namespace base
}  // namespace tachyon
