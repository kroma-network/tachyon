// Copyright (c) 2019 The Color Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/color/color.h"

#include "gtest/gtest.h"

#include "tachyon/base/color/named_color.h"

namespace tachyon::base {

TEST(Rgba, StringConversion) {
  EXPECT_EQ(colors::kGold.ToRgbString(), "rgb(255, 215, 0)");
  EXPECT_EQ(colors::kGold.ToRgbaString(), "rgba(255, 215, 0, 255)");
  EXPECT_EQ(colors::kGold.ToRgbHexString(), "#ffd700");
  EXPECT_EQ(colors::kGold.ToRgbaHexString(), "#ffd700ff");

  Rgba rgba;
  EXPECT_FALSE(rgba.FromString("rgb(255, 215)"));
  EXPECT_FALSE(rgba.FromString("rgb(255, 215, 0"));
  EXPECT_TRUE(rgba.FromString("rgb(255, 215, 0)"));
  EXPECT_EQ(colors::kGold, rgba);
  EXPECT_TRUE(rgba.FromString("rgba(255, 215, 0, 255)"));
  EXPECT_EQ(colors::kGold, rgba);
  EXPECT_FALSE(rgba.FromString("rgb(255, 215, 0, 255"));
  EXPECT_FALSE(rgba.FromString("rgb(255, 215, 0, 255, 0)"));
  EXPECT_FALSE(rgba.FromString("#ffd7"));
  EXPECT_TRUE(rgba.FromString("#ffd700"));
  EXPECT_FALSE(rgba.FromString("#ffd70"));
  EXPECT_EQ(colors::kGold, rgba);
  EXPECT_TRUE(rgba.FromString("#ffd700ff"));
  EXPECT_EQ(colors::kGold, rgba);
  EXPECT_FALSE(rgba.FromString("#ffd700ff0"));
}

TEST(Hsv, StringConversion) {
  Hsv hsv(0.1, 0.2, 0.3, 0.4);
  EXPECT_EQ(hsv.ToHsvString(), "hsv(0.100000, 0.200000, 0.300000)");
  EXPECT_EQ(hsv.ToHsvaString(), "hsva(0.100000, 0.200000, 0.300000, 0.400000)");

  Hsv hsv2;
  EXPECT_TRUE(hsv2.FromString("hsv(0.1, 0.2, 0.3)"));
  EXPECT_EQ(Hsv(0.1, 0.2, 0.3, 1), hsv2);
  EXPECT_TRUE(hsv2.FromString("hsva(0.1, 0.2, 0.3, 0.4)"));
  EXPECT_EQ(hsv, hsv2);
}

}  // namespace tachyon::base
