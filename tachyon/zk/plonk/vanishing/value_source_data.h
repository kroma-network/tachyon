// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_VANISHING_VALUE_SOURCE_DATA_H_
#define TACHYON_ZK_PLONK_VANISHING_VALUE_SOURCE_DATA_H_

#include <stddef.h>

#include <vector>

namespace tachyon::zk {

template <typename Poly>
struct ValueSourceData {
  using F = typename Poly::Field;

  std::vector<size_t> rotations;
  std::vector<F> constants;
  std::vector<F> intermediates;
  std::vector<Poly> fixed_columns;
  std::vector<Poly> advice_columns;
  std::vector<Poly> instance_columns;
  std::vector<F> challenges;
  F beta;
  F gamma;
  F theta;
  F y;
  F previous_value;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_VANISHING_VALUE_SOURCE_DATA_H_
