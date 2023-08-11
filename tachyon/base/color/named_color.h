// Copyright (c) 2019 The Color Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_BASE_COLOR_NAMED_COLOR_H_
#define TACHYON_BASE_COLOR_NAMED_COLOR_H_

#include "tachyon/base/color/color.h"

namespace tachyon::base::colors {
// https://developer.mozilla.org/ko/docs/Web/CSS/color_value

// CSS Level 1
constexpr const Rgba kBlack = Rgba(0, 0, 0);
constexpr const Rgba kSilver = Rgba(192, 192, 192);
constexpr const Rgba kGray = Rgba(128, 128, 128);
constexpr const Rgba kWhite = Rgba(255, 255, 255);
constexpr const Rgba kMaroon = Rgba(128, 0, 0);
constexpr const Rgba kRed = Rgba(255, 0, 0);
constexpr const Rgba kPurple = Rgba(128, 0, 128);
constexpr const Rgba kFuchsia = Rgba(255, 0, 255);
constexpr const Rgba kGreen = Rgba(0, 128, 0);
constexpr const Rgba kLime = Rgba(0, 255, 0);
constexpr const Rgba kOlive = Rgba(128, 128, 0);
constexpr const Rgba kYellow = Rgba(255, 255, 0);
constexpr const Rgba kNavy = Rgba(0, 0, 128);
constexpr const Rgba kBlue = Rgba(0, 0, 255);
constexpr const Rgba kTeal = Rgba(0, 128, 128);
constexpr const Rgba kAqua = Rgba(0, 255, 255);
// CSS Level 2(Revision 1)
constexpr const Rgba kOrange = Rgba(255, 165, 0);
// CSS Color Module Level 3
constexpr const Rgba kAliceblue = Rgba(240, 248, 255);
constexpr const Rgba kAntiquewhite = Rgba(250, 235, 215);
constexpr const Rgba kAquamarine = Rgba(127, 255, 212);
constexpr const Rgba kAzure = Rgba(240, 255, 255);
constexpr const Rgba kBeige = Rgba(245, 245, 220);
constexpr const Rgba kBisque = Rgba(255, 228, 196);
constexpr const Rgba kBlanchedalmond = Rgba(255, 235, 205);
constexpr const Rgba kBlueviolet = Rgba(138, 43, 226);
constexpr const Rgba kBrown = Rgba(165, 42, 42);
constexpr const Rgba kBurlywood = Rgba(222, 184, 135);
constexpr const Rgba kCadetblue = Rgba(95, 158, 160);
constexpr const Rgba kChartreuse = Rgba(127, 255, 0);
constexpr const Rgba kChocolate = Rgba(210, 105, 30);
constexpr const Rgba kCoral = Rgba(255, 127, 80);
constexpr const Rgba kCornflowerblue = Rgba(100, 149, 237);
constexpr const Rgba kCornsilk = Rgba(255, 248, 220);
constexpr const Rgba kCrimson = Rgba(220, 20, 60);
constexpr const Rgba kCyan = Rgba(0, 255, 255);
constexpr const Rgba kDarkblue = Rgba(0, 0, 139);
constexpr const Rgba kDarkcyan = Rgba(0, 139, 139);
constexpr const Rgba kDarkgoldenrod = Rgba(184, 134, 11);
constexpr const Rgba kDarkgray = Rgba(169, 169, 169);
constexpr const Rgba kDarkgreen = Rgba(0, 100, 0);
constexpr const Rgba kDarkgrey = Rgba(169, 169, 169);
constexpr const Rgba kDarkkhaki = Rgba(189, 183, 107);
constexpr const Rgba kDarkmagenta = Rgba(139, 0, 139);
constexpr const Rgba kDarkolivegreen = Rgba(85, 107, 47);
constexpr const Rgba kDarkorange = Rgba(255, 140, 0);
constexpr const Rgba kDarkorchid = Rgba(153, 50, 204);
constexpr const Rgba kDarkred = Rgba(139, 0, 0);
constexpr const Rgba kDarksalmon = Rgba(233, 150, 122);
constexpr const Rgba kDarkseagreen = Rgba(143, 188, 143);
constexpr const Rgba kDarkslateblue = Rgba(72, 61, 139);
constexpr const Rgba kDarkslategray = Rgba(47, 79, 79);
constexpr const Rgba kDarkslategrey = Rgba(47, 79, 79);
constexpr const Rgba kDarkturquoise = Rgba(0, 206, 209);
constexpr const Rgba kDarkviolet = Rgba(148, 0, 211);
constexpr const Rgba kDeeppink = Rgba(255, 20, 147);
constexpr const Rgba kDeepskyblue = Rgba(0, 191, 255);
constexpr const Rgba kDimgray = Rgba(105, 105, 105);
constexpr const Rgba kDimgrey = Rgba(105, 105, 105);
constexpr const Rgba kDodgerblue = Rgba(30, 144, 255);
constexpr const Rgba kFirebrick = Rgba(178, 34, 34);
constexpr const Rgba kFloralwhite = Rgba(255, 250, 240);
constexpr const Rgba kForestgreen = Rgba(34, 139, 34);
constexpr const Rgba kGainsboro = Rgba(220, 220, 220);
constexpr const Rgba kGhostwhite = Rgba(248, 248, 255);
constexpr const Rgba kGold = Rgba(255, 215, 0);
constexpr const Rgba kGoldenrod = Rgba(218, 165, 32);
constexpr const Rgba kGreenyellow = Rgba(173, 255, 47);
constexpr const Rgba kGrey = Rgba(128, 128, 128);
constexpr const Rgba kHoneydew = Rgba(240, 255, 240);
constexpr const Rgba kHotpink = Rgba(255, 105, 180);
constexpr const Rgba kIndianred = Rgba(205, 92, 92);
constexpr const Rgba kIndigo = Rgba(75, 0, 130);
constexpr const Rgba kIvory = Rgba(255, 255, 240);
constexpr const Rgba kKhaki = Rgba(240, 230, 140);
constexpr const Rgba kLavender = Rgba(230, 230, 250);
constexpr const Rgba kLavenderblush = Rgba(255, 240, 245);
constexpr const Rgba kLawngreen = Rgba(124, 252, 0);
constexpr const Rgba kLemonchiffon = Rgba(255, 250, 205);
constexpr const Rgba kLightblue = Rgba(173, 216, 230);
constexpr const Rgba kLightcoral = Rgba(240, 128, 128);
constexpr const Rgba kLightcyan = Rgba(224, 255, 255);
constexpr const Rgba kLightgoldenrodyellow = Rgba(250, 250, 210);
constexpr const Rgba kLightgray = Rgba(211, 211, 211);
constexpr const Rgba kLightgreen = Rgba(144, 238, 144);
constexpr const Rgba kLightgrey = Rgba(211, 211, 211);
constexpr const Rgba kLightpink = Rgba(255, 182, 193);
constexpr const Rgba kLightsalmon = Rgba(255, 160, 122);
constexpr const Rgba kLightseagreen = Rgba(32, 178, 170);
constexpr const Rgba kLightskyblue = Rgba(135, 206, 250);
constexpr const Rgba kLightslategray = Rgba(119, 136, 153);
constexpr const Rgba kLightslategrey = Rgba(119, 136, 153);
constexpr const Rgba kLightsteelblue = Rgba(176, 196, 222);
constexpr const Rgba kLightyellow = Rgba(255, 255, 224);
constexpr const Rgba kLimegreen = Rgba(50, 205, 50);
constexpr const Rgba kLinen = Rgba(250, 240, 230);
constexpr const Rgba kMagenta = Rgba(255, 0, 255);
constexpr const Rgba kMediumaquamarine = Rgba(102, 205, 170);
constexpr const Rgba kMediumblue = Rgba(0, 0, 205);
constexpr const Rgba kMediumorchid = Rgba(186, 85, 211);
constexpr const Rgba kMediumpurple = Rgba(147, 112, 219);
constexpr const Rgba kMediumseagreen = Rgba(60, 179, 113);
constexpr const Rgba kMediumslateblue = Rgba(123, 104, 238);
constexpr const Rgba kMediumspringgreen = Rgba(0, 250, 154);
constexpr const Rgba kMediumturquoise = Rgba(72, 209, 204);
constexpr const Rgba kMediumvioletred = Rgba(199, 21, 133);
constexpr const Rgba kMidnightblue = Rgba(25, 25, 112);
constexpr const Rgba kMintcream = Rgba(245, 255, 250);
constexpr const Rgba kMistyrose = Rgba(255, 228, 225);
constexpr const Rgba kMoccasin = Rgba(255, 228, 181);
constexpr const Rgba kNavajowhite = Rgba(255, 222, 173);
constexpr const Rgba kOldlace = Rgba(253, 245, 230);
constexpr const Rgba kOlivedrab = Rgba(107, 142, 35);
constexpr const Rgba kOrangered = Rgba(255, 69, 0);
constexpr const Rgba kOrchid = Rgba(218, 112, 214);
constexpr const Rgba kPalegoldenrod = Rgba(238, 232, 170);
constexpr const Rgba kPalegreen = Rgba(152, 251, 152);
constexpr const Rgba kPaleturquoise = Rgba(175, 238, 238);
constexpr const Rgba kPalevioletred = Rgba(219, 112, 147);
constexpr const Rgba kPapayawhip = Rgba(255, 239, 213);
constexpr const Rgba kPeachpuff = Rgba(255, 218, 185);
constexpr const Rgba kPeru = Rgba(205, 133, 63);
constexpr const Rgba kPink = Rgba(255, 192, 203);
constexpr const Rgba kPlum = Rgba(221, 160, 221);
constexpr const Rgba kPowderblue = Rgba(176, 224, 230);
constexpr const Rgba kRosybrown = Rgba(188, 143, 143);
constexpr const Rgba kRoyalblue = Rgba(65, 105, 225);
constexpr const Rgba kSaddlebrown = Rgba(139, 69, 19);
constexpr const Rgba kSalmon = Rgba(250, 128, 114);
constexpr const Rgba kSandybrown = Rgba(244, 164, 96);
constexpr const Rgba kSeagreen = Rgba(46, 139, 87);
constexpr const Rgba kSeashell = Rgba(255, 245, 238);
constexpr const Rgba kSienna = Rgba(160, 82, 45);
constexpr const Rgba kSkyblue = Rgba(135, 206, 235);
constexpr const Rgba kSlateblue = Rgba(106, 90, 205);
constexpr const Rgba kSlategray = Rgba(112, 128, 144);
constexpr const Rgba kSlategrey = Rgba(112, 128, 144);
constexpr const Rgba kSnow = Rgba(255, 250, 250);
constexpr const Rgba kSpringgreen = Rgba(0, 255, 127);
constexpr const Rgba kSteelblue = Rgba(70, 130, 180);
constexpr const Rgba kTan = Rgba(210, 180, 140);
constexpr const Rgba kThistle = Rgba(216, 191, 216);
constexpr const Rgba kTomato = Rgba(255, 99, 71);
constexpr const Rgba kTurquoise = Rgba(64, 224, 208);
constexpr const Rgba kViolet = Rgba(238, 130, 238);
constexpr const Rgba kWheat = Rgba(245, 222, 179);
constexpr const Rgba kWhitesmoke = Rgba(245, 245, 245);
constexpr const Rgba kYellowgreen = Rgba(154, 205, 50);
// CSS Color Module Level 4
constexpr const Rgba kRebeccapurple = Rgba(102, 51, 153);

}  // namespace tachyon::base::colors

#endif  // TACHYON_BASE_COLOR_NAMED_COLOR_H_
