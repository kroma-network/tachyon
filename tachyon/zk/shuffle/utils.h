// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_SHUFFLE_UTILS_H_
#define TACHYON_ZK_SHUFFLE_UTILS_H_

#include <stddef.h>

namespace tachyon::zk::shuffle {

constexpr size_t GetNumEvals(size_t num_circuits, size_t num_shuffles) {
  return num_circuits * num_shuffles * 3;
}

constexpr size_t GetNumOpenings(size_t num_circuits, size_t num_shuffles) {
  return num_circuits * num_shuffles * 2;
}

}  // namespace tachyon::zk::shuffle

#endif  // TACHYON_ZK_SHUFFLE_UTILS_H_
