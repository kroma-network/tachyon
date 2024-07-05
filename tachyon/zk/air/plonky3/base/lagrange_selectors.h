// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_ZK_AIR_PLONKY3_BASE_LAGRANGE_SELECTORS_H_
#define TACHYON_ZK_AIR_PLONKY3_BASE_LAGRANGE_SELECTORS_H_

namespace tachyon::zk::air::plonky3 {

template <typename T>
struct LagrangeSelectors {
  T first_row;
  T last_row;
  T transition;
  T inv_zeroifier;

  bool operator==(const LagrangeSelectors& other) const {
    return first_row == other.first_row && last_row == other.last_row &&
           transition == other.transition &&
           inv_zeroifier == other.inv_zeroifier;
  }
  bool operator!=(const LagrangeSelectors& other) const {
    return !operator==(other);
  }
};

}  // namespace tachyon::zk::air::plonky3

#endif  // TACHYON_ZK_AIR_PLONKY3_BASE_LAGRANGE_SELECTORS_H_
