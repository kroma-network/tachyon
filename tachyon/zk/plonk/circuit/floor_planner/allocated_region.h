// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_ALLOCATED_REGION_H_
#define TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_ALLOCATED_REGION_H_

#include <stddef.h>

#include "tachyon/export.h"

namespace tachyon::zk {

// A region allocated within a column.
class TACHYON_EXPORT AllocatedRegion {
 public:
  constexpr AllocatedRegion(size_t start, size_t length)
      : start_(start), length_(length) {}

  constexpr size_t start() const { return start_; }
  constexpr size_t length() const { return length_; }

  constexpr size_t End() const { return start_ + length_; }

  constexpr bool operator<(const AllocatedRegion& other) const {
    return start_ < other.start_;
  }

 private:
  // The starting position of the region.
  size_t start_ = 0;
  // The length of the region.
  size_t length_ = 0;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_ALLOCATED_REGION_H_
