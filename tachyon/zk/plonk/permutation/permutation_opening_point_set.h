// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_OPENING_POINT_SET_H_
#define TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_OPENING_POINT_SET_H_

namespace tachyon::zk::plonk {

template <typename F>
struct PermutationOpeningPointSet {
  PermutationOpeningPointSet(const F& x, const F& x_next, const F& x_last)
      : x(x), x_next(x_next), x_last(x_last) {}

  const F& x;
  const F& x_next;
  const F& x_last;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_OPENING_POINT_SET_H_
