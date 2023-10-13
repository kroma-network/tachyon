// Copyright 2014 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/containers/adapters.h"

#include <vector>

#include "gtest/gtest.h"

namespace tachyon::base {

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

template <typename V, typename T>
void TestChunked(T&& v) {
  for (size_t chunk_size = 1; chunk_size < 5; ++chunk_size) {
    size_t expected_num_chunks = (std::size(v) + chunk_size - 1) / chunk_size;
    size_t chunk_offset = 0;
    size_t offset = 0;
    size_t size = std::size(v);
    for (const absl::Span<V>& j : Chunked(std::forward<T>(v), chunk_size)) {
      EXPECT_EQ(*j.data(), v[offset]);
      EXPECT_EQ(j.size(), chunk_offset == expected_num_chunks - 1
                              ? size - offset
                              : chunk_size);
      ++chunk_offset;
      offset += chunk_size;
    }
    EXPECT_EQ(chunk_offset, expected_num_chunks);
  }
}

TEST(AdaptersTest, Chunked) {
  std::vector<int> v;
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);
  TestChunked<int>(v);
}

TEST(AdaptersTest, ChunkedArray) {
  int v[3] = {1, 2, 3};
  TestChunked<int>(v);
}

TEST(AdaptersTest, ChunkedConst) {
  std::vector<int> v;
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);
  const std::vector<int>& cv = v;
  TestChunked<const int>(cv);
}

TEST(AdaptersTest, Zipped) {
  std::vector<int> v;
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);
  std::vector<int> w;
  w.push_back(4);
  w.push_back(5);
  w.push_back(6);
  size_t offset = 0;
  for (const std::tuple<int, int>& j : Zipped(v, w)) {
    EXPECT_EQ(std::get<0>(j), v[offset]);
    EXPECT_EQ(std::get<1>(j), w[offset]);
    ++offset;
  }
  EXPECT_EQ(offset, std::size(v));
}

TEST(AdaptersTest, ZippedArray) {
  int v[3] = {1, 2, 3};
  int w[3] = {4, 5, 6};
  size_t offset = 0;
  for (const std::tuple<int, int>& j : Zipped(v, w)) {
    EXPECT_EQ(std::get<0>(j), v[offset]);
    EXPECT_EQ(std::get<1>(j), w[offset]);
    ++offset;
  }
  EXPECT_EQ(offset, std::size(v));
}

TEST(AdaptersTest, ZippedConst) {
  std::vector<int> v;
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);
  std::vector<int> w;
  w.push_back(4);
  w.push_back(5);
  w.push_back(6);
  const std::vector<int>& cv = v;
  const std::vector<int>& cw = w;
  size_t offset = 0;
  for (const std::tuple<int, int>& j : Zipped(cv, cw)) {
    EXPECT_EQ(std::get<0>(j), v[offset]);
    EXPECT_EQ(std::get<1>(j), w[offset]);
    ++offset;
  }
  EXPECT_EQ(offset, std::size(v));
}

}  // namespace tachyon::base
