// Copyright 2019 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/test/copy_only_int.h"

namespace tachyon::base {

// static
int CopyOnlyInt::num_copies_ = 0;

}  // namespace tachyon::base
