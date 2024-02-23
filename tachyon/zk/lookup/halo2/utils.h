// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_LOOKUP_HALO2_UTILS_H_
#define TACHYON_ZK_LOOKUP_HALO2_UTILS_H_

#include <stddef.h>

namespace tachyon::zk::lookup::halo2 {

constexpr size_t GetNumEvals(size_t num_circuits, size_t num_lookups) {
  return num_circuits * num_lookups * 5;
}

constexpr size_t GetNumOpenings(size_t num_circuits, size_t num_lookups) {
  return num_circuits * num_lookups * 5;
}

}  // namespace tachyon::zk::lookup::halo2

#endif  // TACHYON_ZK_LOOKUP_HALO2_UTILS_H_
