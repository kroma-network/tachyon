// Copyright 2012 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/memory/aligned_memory.h"

#include <memory>

#include "gtest/gtest.h"

namespace tachyon::base {

TEST(AlignedMemoryTest, DynamicAllocation) {
  void* p = AlignedAlloc(8, 8);
  EXPECT_TRUE(p);
  EXPECT_TRUE(IsAligned(p, 8));
  AlignedFree(p);

  p = AlignedAlloc(8, 16);
  EXPECT_TRUE(p);
  EXPECT_TRUE(IsAligned(p, 16));
  AlignedFree(p);

  p = AlignedAlloc(8, 256);
  EXPECT_TRUE(p);
  EXPECT_TRUE(IsAligned(p, 256));
  AlignedFree(p);

  p = AlignedAlloc(8, 4096);
  EXPECT_TRUE(p);
  EXPECT_TRUE(IsAligned(p, 4096));
  AlignedFree(p);
}

TEST(AlignedMemoryTest, ScopedDynamicAllocation) {
  std::unique_ptr<float, AlignedFreeDeleter> p(
      static_cast<float*>(AlignedAlloc(8, 8)));
  EXPECT_TRUE(p.get());
  EXPECT_TRUE(IsAligned(p.get(), 8));

  // Make sure IsAligned() can check const pointers as well.
  const float* const_p = p.get();
  EXPECT_TRUE(IsAligned(const_p, 8));
}

TEST(AlignedMemoryTest, IsAligned) {
  // Check alignment around powers of two.
  for (int i = 0; i < 64; ++i) {
    const uint64_t n = static_cast<uint64_t>(1) << i;

    // Walk back down all lower powers of two checking alignment.
    for (int j = i - 1; j >= 0; --j) {
      // n is aligned on all powers of two less than or equal to 2^i.
      EXPECT_TRUE(IsAligned(n, n >> j))
          << "Expected " << n << " to be " << (n >> j) << " aligned";

      // Also, n - 1 should not be aligned on ANY lower power of two except 1
      // (but since we're starting from i - 1 we don't test that case here.
      EXPECT_FALSE(IsAligned(n - 1, n >> j))
          << "Expected " << (n - 1) << " to NOT be " << (n >> j) << " aligned";
    }
  }

  // And a few hard coded smoke tests for completeness:
  EXPECT_TRUE(IsAligned(4, 2));
  EXPECT_TRUE(IsAligned(8, 4));
  EXPECT_TRUE(IsAligned(8, 2));
  EXPECT_TRUE(IsAligned(0x1000, 4 << 10));
  EXPECT_TRUE(IsAligned(0x2000, 8 << 10));
  EXPECT_TRUE(IsAligned(1, 1));
  EXPECT_TRUE(IsAligned(7, 1));
  EXPECT_TRUE(IsAligned(reinterpret_cast<void*>(0x1000), 4 << 10));
  EXPECT_TRUE(IsAligned(reinterpret_cast<int*>(0x1000), 4 << 10));

  EXPECT_FALSE(IsAligned(3, 2));
  EXPECT_FALSE(IsAligned(7, 4));
  EXPECT_FALSE(IsAligned(7, 2));
  EXPECT_FALSE(IsAligned(0x1001, 4 << 10));
  EXPECT_FALSE(IsAligned(0x999, 8 << 10));
  EXPECT_FALSE(IsAligned(7, 8));
}

}  // namespace tachyon::base
