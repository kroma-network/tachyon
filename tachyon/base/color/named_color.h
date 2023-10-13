// Copyright (c) 2019 The Color Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_BASE_COLOR_NAMED_COLOR_H_
#define TACHYON_BASE_COLOR_NAMED_COLOR_H_

#include "tachyon/base/color/color.h"

namespace tachyon::base::colors {
// https://developer.mozilla.org/ko/docs/Web/CSS/color_value

// CSS Level 1
constexpr Rgba kBlack = Rgba(0, 0, 0);
constexpr Rgba kSilver = Rgba(192, 192, 192);
constexpr Rgba kGray = Rgba(128, 128, 128);
constexpr Rgba kWhite = Rgba(255, 255, 255);
constexpr Rgba kMaroon = Rgba(128, 0, 0);
constexpr Rgba kRed = Rgba(255, 0, 0);
constexpr Rgba kPurple = Rgba(128, 0, 128);
constexpr Rgba kFuchsia = Rgba(255, 0, 255);
constexpr Rgba kGreen = Rgba(0, 128, 0);
constexpr Rgba kLime = Rgba(0, 255, 0);
constexpr Rgba kOlive = Rgba(128, 128, 0);
constexpr Rgba kYellow = Rgba(255, 255, 0);
constexpr Rgba kNavy = Rgba(0, 0, 128);
constexpr Rgba kBlue = Rgba(0, 0, 255);
constexpr Rgba kTeal = Rgba(0, 128, 128);
constexpr Rgba kAqua = Rgba(0, 255, 255);
// CSS Level 2(Revision 1)
constexpr Rgba kOrange = Rgba(255, 165, 0);
// CSS Color Module Level 3
constexpr Rgba kAliceblue = Rgba(240, 248, 255);
constexpr Rgba kAntiquewhite = Rgba(250, 235, 215);
constexpr Rgba kAquamarine = Rgba(127, 255, 212);
constexpr Rgba kAzure = Rgba(240, 255, 255);
constexpr Rgba kBeige = Rgba(245, 245, 220);
constexpr Rgba kBisque = Rgba(255, 228, 196);
constexpr Rgba kBlanchedalmond = Rgba(255, 235, 205);
constexpr Rgba kBlueviolet = Rgba(138, 43, 226);
constexpr Rgba kBrown = Rgba(165, 42, 42);
constexpr Rgba kBurlywood = Rgba(222, 184, 135);
constexpr Rgba kCadetblue = Rgba(95, 158, 160);
constexpr Rgba kChartreuse = Rgba(127, 255, 0);
constexpr Rgba kChocolate = Rgba(210, 105, 30);
constexpr Rgba kCoral = Rgba(255, 127, 80);
constexpr Rgba kCornflowerblue = Rgba(100, 149, 237);
constexpr Rgba kCornsilk = Rgba(255, 248, 220);
constexpr Rgba kCrimson = Rgba(220, 20, 60);
constexpr Rgba kCyan = Rgba(0, 255, 255);
constexpr Rgba kDarkblue = Rgba(0, 0, 139);
constexpr Rgba kDarkcyan = Rgba(0, 139, 139);
constexpr Rgba kDarkgoldenrod = Rgba(184, 134, 11);
constexpr Rgba kDarkgray = Rgba(169, 169, 169);
constexpr Rgba kDarkgreen = Rgba(0, 100, 0);
constexpr Rgba kDarkgrey = Rgba(169, 169, 169);
constexpr Rgba kDarkkhaki = Rgba(189, 183, 107);
constexpr Rgba kDarkmagenta = Rgba(139, 0, 139);
constexpr Rgba kDarkolivegreen = Rgba(85, 107, 47);
constexpr Rgba kDarkorange = Rgba(255, 140, 0);
constexpr Rgba kDarkorchid = Rgba(153, 50, 204);
constexpr Rgba kDarkred = Rgba(139, 0, 0);
constexpr Rgba kDarksalmon = Rgba(233, 150, 122);
constexpr Rgba kDarkseagreen = Rgba(143, 188, 143);
constexpr Rgba kDarkslateblue = Rgba(72, 61, 139);
constexpr Rgba kDarkslategray = Rgba(47, 79, 79);
constexpr Rgba kDarkslategrey = Rgba(47, 79, 79);
constexpr Rgba kDarkturquoise = Rgba(0, 206, 209);
constexpr Rgba kDarkviolet = Rgba(148, 0, 211);
constexpr Rgba kDeeppink = Rgba(255, 20, 147);
constexpr Rgba kDeepskyblue = Rgba(0, 191, 255);
constexpr Rgba kDimgray = Rgba(105, 105, 105);
constexpr Rgba kDimgrey = Rgba(105, 105, 105);
constexpr Rgba kDodgerblue = Rgba(30, 144, 255);
constexpr Rgba kFirebrick = Rgba(178, 34, 34);
constexpr Rgba kFloralwhite = Rgba(255, 250, 240);
constexpr Rgba kForestgreen = Rgba(34, 139, 34);
constexpr Rgba kGainsboro = Rgba(220, 220, 220);
constexpr Rgba kGhostwhite = Rgba(248, 248, 255);
constexpr Rgba kGold = Rgba(255, 215, 0);
constexpr Rgba kGoldenrod = Rgba(218, 165, 32);
constexpr Rgba kGreenyellow = Rgba(173, 255, 47);
constexpr Rgba kGrey = Rgba(128, 128, 128);
constexpr Rgba kHoneydew = Rgba(240, 255, 240);
constexpr Rgba kHotpink = Rgba(255, 105, 180);
constexpr Rgba kIndianred = Rgba(205, 92, 92);
constexpr Rgba kIndigo = Rgba(75, 0, 130);
constexpr Rgba kIvory = Rgba(255, 255, 240);
constexpr Rgba kKhaki = Rgba(240, 230, 140);
constexpr Rgba kLavender = Rgba(230, 230, 250);
constexpr Rgba kLavenderblush = Rgba(255, 240, 245);
constexpr Rgba kLawngreen = Rgba(124, 252, 0);
constexpr Rgba kLemonchiffon = Rgba(255, 250, 205);
constexpr Rgba kLightblue = Rgba(173, 216, 230);
constexpr Rgba kLightcoral = Rgba(240, 128, 128);
constexpr Rgba kLightcyan = Rgba(224, 255, 255);
constexpr Rgba kLightgoldenrodyellow = Rgba(250, 250, 210);
constexpr Rgba kLightgray = Rgba(211, 211, 211);
constexpr Rgba kLightgreen = Rgba(144, 238, 144);
constexpr Rgba kLightgrey = Rgba(211, 211, 211);
constexpr Rgba kLightpink = Rgba(255, 182, 193);
constexpr Rgba kLightsalmon = Rgba(255, 160, 122);
constexpr Rgba kLightseagreen = Rgba(32, 178, 170);
constexpr Rgba kLightskyblue = Rgba(135, 206, 250);
constexpr Rgba kLightslategray = Rgba(119, 136, 153);
constexpr Rgba kLightslategrey = Rgba(119, 136, 153);
constexpr Rgba kLightsteelblue = Rgba(176, 196, 222);
constexpr Rgba kLightyellow = Rgba(255, 255, 224);
constexpr Rgba kLimegreen = Rgba(50, 205, 50);
constexpr Rgba kLinen = Rgba(250, 240, 230);
constexpr Rgba kMagenta = Rgba(255, 0, 255);
constexpr Rgba kMediumaquamarine = Rgba(102, 205, 170);
constexpr Rgba kMediumblue = Rgba(0, 0, 205);
constexpr Rgba kMediumorchid = Rgba(186, 85, 211);
constexpr Rgba kMediumpurple = Rgba(147, 112, 219);
constexpr Rgba kMediumseagreen = Rgba(60, 179, 113);
constexpr Rgba kMediumslateblue = Rgba(123, 104, 238);
constexpr Rgba kMediumspringgreen = Rgba(0, 250, 154);
constexpr Rgba kMediumturquoise = Rgba(72, 209, 204);
constexpr Rgba kMediumvioletred = Rgba(199, 21, 133);
constexpr Rgba kMidnightblue = Rgba(25, 25, 112);
constexpr Rgba kMintcream = Rgba(245, 255, 250);
constexpr Rgba kMistyrose = Rgba(255, 228, 225);
constexpr Rgba kMoccasin = Rgba(255, 228, 181);
constexpr Rgba kNavajowhite = Rgba(255, 222, 173);
constexpr Rgba kOldlace = Rgba(253, 245, 230);
constexpr Rgba kOlivedrab = Rgba(107, 142, 35);
constexpr Rgba kOrangered = Rgba(255, 69, 0);
constexpr Rgba kOrchid = Rgba(218, 112, 214);
constexpr Rgba kPalegoldenrod = Rgba(238, 232, 170);
constexpr Rgba kPalegreen = Rgba(152, 251, 152);
constexpr Rgba kPaleturquoise = Rgba(175, 238, 238);
constexpr Rgba kPalevioletred = Rgba(219, 112, 147);
constexpr Rgba kPapayawhip = Rgba(255, 239, 213);
constexpr Rgba kPeachpuff = Rgba(255, 218, 185);
constexpr Rgba kPeru = Rgba(205, 133, 63);
constexpr Rgba kPink = Rgba(255, 192, 203);
constexpr Rgba kPlum = Rgba(221, 160, 221);
constexpr Rgba kPowderblue = Rgba(176, 224, 230);
constexpr Rgba kRosybrown = Rgba(188, 143, 143);
constexpr Rgba kRoyalblue = Rgba(65, 105, 225);
constexpr Rgba kSaddlebrown = Rgba(139, 69, 19);
constexpr Rgba kSalmon = Rgba(250, 128, 114);
constexpr Rgba kSandybrown = Rgba(244, 164, 96);
constexpr Rgba kSeagreen = Rgba(46, 139, 87);
constexpr Rgba kSeashell = Rgba(255, 245, 238);
constexpr Rgba kSienna = Rgba(160, 82, 45);
constexpr Rgba kSkyblue = Rgba(135, 206, 235);
constexpr Rgba kSlateblue = Rgba(106, 90, 205);
constexpr Rgba kSlategray = Rgba(112, 128, 144);
constexpr Rgba kSlategrey = Rgba(112, 128, 144);
constexpr Rgba kSnow = Rgba(255, 250, 250);
constexpr Rgba kSpringgreen = Rgba(0, 255, 127);
constexpr Rgba kSteelblue = Rgba(70, 130, 180);
constexpr Rgba kTan = Rgba(210, 180, 140);
constexpr Rgba kThistle = Rgba(216, 191, 216);
constexpr Rgba kTomato = Rgba(255, 99, 71);
constexpr Rgba kTurquoise = Rgba(64, 224, 208);
constexpr Rgba kViolet = Rgba(238, 130, 238);
constexpr Rgba kWheat = Rgba(245, 222, 179);
constexpr Rgba kWhitesmoke = Rgba(245, 245, 245);
constexpr Rgba kYellowgreen = Rgba(154, 205, 50);
// CSS Color Module Level 4
constexpr Rgba kRebeccapurple = Rgba(102, 51, 153);

}  // namespace tachyon::base::colors

#endif  // TACHYON_BASE_COLOR_NAMED_COLOR_H_
