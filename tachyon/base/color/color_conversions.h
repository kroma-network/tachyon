// Copyright (c) 2019 The Color Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_BASE_COLOR_COLOR_CONVERSIONS_H_
#define TACHYON_BASE_COLOR_COLOR_CONVERSIONS_H_

#include "tachyon/base/color/color.h"
#include "tachyon/export.h"

namespace tachyon::base {

TACHYON_EXPORT Rgba HsvToRgba(const Hsv& hsv);

TACHYON_EXPORT Hsv RgbaToHsv(Rgba rgba);

}  // namespace tachyon::base

#endif  // TACHYON_BASE_COLOR_COLOR_CONVERSIONS_H_
