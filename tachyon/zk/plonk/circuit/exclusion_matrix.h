// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_EXCLUSION_MATRIX_H_
#define TACHYON_ZK_PLONK_CIRCUIT_EXCLUSION_MATRIX_H_

#include <algorithm>
#include <utility>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/export.h"
#include "tachyon/zk/plonk/circuit/selector_description.h"

namespace tachyon::zk {

// Exclusion matrix that has (i, j) = true if activations vector of selector i
// and selector j conflict -- that is, they are both enabled on the same row.
// This matrix is symmetric and the diagonal entries are false, so we only need
// to store the lower triangular entries.
//
// For example, given the following selectors:
//   [1, 0, 0, 0, 0, 0, 0, 0, 1]
//   [1, 0, 0, 0, 0, 0, 0, 1, 0]
//   [1, 0, 0, 0, 0, 0, 1, 0, 0]
//   [0, 1, 0, 0, 0, 1, 1, 1, 0]
//   [0, 1, 0, 0, 1, 0, 1, 0, 1]
//   [0, 1, 0, 1, 0, 0, 0, 1, 1]
//   [0, 0, 1, 1, 1, 0, 0, 0, 0]
//   [0, 0, 1, 1, 0, 1, 0, 0, 0]
//   [0, 0, 1, 0, 1, 1, 0, 0, 0]
//
// Exclusion matrix is the lower triangular entries
// of the following matrix:
// i \ j |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |
// ------+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//   0   |  0  |  1  |  1  |  0  |  1  |  1  |  0  |  0  |  0  |
// ------+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//   1   |  1  |  0  |  1  |  1  |  0  |  1  |  0  |  0  |  0  |
// ------+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//   2   |  1  |  1  |  0  |  1  |  1  |  0  |  0  |  0  |  0  |
// ------+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//   3   |  0  |  1  |  1  |  0  |  1  |  1  |  0  |  1  |  1  |
// ------+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//   4   |  1  |  0  |  1  |  1  |  0  |  1  |  1  |  0  |  1  |
// ------+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//   5   |  1  |  1  |  0  |  1  |  1  |  0  |  1  |  1  |  0  |
// ------+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//   6   |  0  |  0  |  0  |  0  |  1  |  1  |  0  |  1  |  1  |
// ------+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//   7   |  0  |  0  |  0  |  1  |  0  |  1  |  1  |  0  |  1  |
// ------+-----+-----+-----+-----+-----+-----+-----+-----+-----+
//   8   |  0  |  0  |  0  |  1  |  1  |  0  |  1  |  1  |  0  |
// ------+-----+-----+-----+-----+-----+-----+-----+-----+-----+

class TACHYON_EXPORT ExclusionMatrix {
 public:
  explicit ExclusionMatrix(const std::vector<SelectorDescription>& selectors) {
    lower_triangular_matrix_ =
        base::CreateVector(selectors.size(), [&selectors](size_t i) {
          const SelectorDescription& selector = selectors[i];
          return base::CreateVector(i, [&selector, &selectors](size_t j) {
            return !selector.IsOrthogonal(selectors[j]);
          });
        });
  }

  const std::vector<std::vector<bool>>& lower_triangular_matrix() const {
    return lower_triangular_matrix_;
  }

  bool IsExclusive(size_t src, size_t dst) const {
    size_t i = std::max(src, dst);
    size_t j = std::min(src, dst);
    return lower_triangular_matrix_[i][j];
  }

 private:
  std::vector<std::vector<bool>> lower_triangular_matrix_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_EXCLUSION_MATRIX_H_
