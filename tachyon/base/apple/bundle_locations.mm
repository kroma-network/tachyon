// Copyright 2012 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/apple/bundle_locations.h"

#if !defined(__has_feature) || !__has_feature(objc_arc)
#error "This file requires ARC support."
#endif

namespace tachyon::base::apple {

NSBundle* MainBundle() {
  return NSBundle.mainBundle;
}

NSURL* MainBundleURL() {
  return MainBundle().bundleURL;
}

NSURL* OuterBundleURL() {
  return OuterBundle().bundleURL;
}

}  // namespace tachyon::base::apple
