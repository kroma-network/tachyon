// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_ARGUMENT_H_
#define TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_ARGUMENT_H_

#include <utility>
#include <vector>

#include "tachyon/base/containers/contains.h"
#include "tachyon/zk/plonk/circuit/column.h"

namespace tachyon::zk {

// TODO(dongchangYoo): impl logics related to prove and verify.
class PermutationArgument {
 public:
  PermutationArgument() = default;
  explicit PermutationArgument(const std::vector<AnyColumn>& columns)
      : columns_(columns) {}
  explicit PermutationArgument(std::vector<AnyColumn>&& columns)
      : columns_(std::move(columns)) {}

  void AddColumn(const AnyColumn& column) {
    if (base::Contains(columns_, column)) return;
    columns_.push_back(column);
  }

  const std::vector<AnyColumn>& columns() const { return columns_; }

 private:
  std::vector<AnyColumn> columns_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_ARGUMENT_H_
