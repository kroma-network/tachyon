// Copyright 2018 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_BASE_POSIX_CAN_LOWER_NICE_TO_H_
#define TACHYON_BASE_POSIX_CAN_LOWER_NICE_TO_H_

namespace tachyon::base::internal {

// Returns true if lowering the nice value of a process or thread to
// |nice_value| using setpriority() or nice() should succeed. Note: A lower nice
// value means a higher priority.
bool CanLowerNiceTo(int nice_value);

}  // namespace tachyon::base::internal

#endif  // TACHYON_BASE_POSIX_CAN_LOWER_NICE_TO_H_
