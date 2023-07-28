// Copyright 2012 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_BASE_MEMORY_SCOPED_POLICY_H_
#define TACHYON_BASE_MEMORY_SCOPED_POLICY_H_

namespace tachyon::base {
namespace scoped_policy {

// Defines the ownership policy for a scoped object.
enum OwnershipPolicy {
  // The scoped object takes ownership of an object by taking over an existing
  // ownership claim.
  ASSUME,

  // The scoped object will retain the object and any initial ownership is
  // not changed.
  RETAIN
};

}  // namespace scoped_policy
}  // namespace tachyon::base

#endif  // TACHYON_BASE_MEMORY_SCOPED_POLICY_H_
