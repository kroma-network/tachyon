// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_SHUFFLE_OPENING_POINT_SET_H_
#define TACHYON_ZK_SHUFFLE_OPENING_POINT_SET_H_

namespace tachyon::zk::shuffle {

template <typename F>
struct OpeningPointSet {
  OpeningPointSet(const F& x, const F& x_next) : x(x), x_next(x_next) {}

  const F& x;
  const F& x_next;
};

}  // namespace tachyon::zk::shuffle

#endif  // TACHYON_ZK_SHUFFLE_OPENING_POINT_SET_H_
