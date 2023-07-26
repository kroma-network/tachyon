// Copyright (c) 2019 The Color Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/color/color_conversions.h"

#include "gtest/gtest.h"

#include "tachyon/base/color/named_color.h"

namespace tachyon {
namespace base {

TEST(ColorConversions, RgbaAndHsv) {
  Hsv hsv = RgbaToHsv(colors::kMagenta);
  EXPECT_EQ(300, hsv.h);
  EXPECT_EQ(1, hsv.s);
  EXPECT_EQ(1, hsv.v);
  Rgba rgba = HsvToRgba(hsv);
  EXPECT_EQ(colors::kMagenta, rgba);
}

}  // namespace base
}  // namespace tachyon
