// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_PERMUTATION_PERMUTED_TABLE_H_
#define TACHYON_ZK_PLONK_PERMUTATION_PERMUTED_TABLE_H_

#include <utility>
#include <vector>

#include "tachyon/base/range.h"
#include "tachyon/base/ref.h"
#include "tachyon/zk/plonk/permutation/label.h"

namespace tachyon::zk {

template <typename Evals>
class PermutedTable {
 public:
  using F = typename Evals::Field;
  using Table = std::vector<Evals>;

  PermutedTable() = default;
  explicit PermutedTable(const Table* table) : table_(table) {}

  const F& operator[](const Label& label) const {
    return (*table_)[label.col][label.row];
  }

  base::Ref<const Evals> GetColumn(size_t i) const {
    return base::Ref<const Evals>(&(*table_)[i]);
  }

  std::vector<base::Ref<const Evals>> GetColumns(
      base::Range<size_t> range) const {
    CHECK_EQ(range.Intersect(base::Range<size_t>::Until(table_->size())),
             range);

    std::vector<base::Ref<const Evals>> ret;
    ret.reserve(range.GetSize());
    for (size_t i : range) {
      ret.push_back(GetColumn(i));
    }
    return ret;
  }

 private:
  // not owned
  const Table* table_ = nullptr;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_PERMUTED_TABLE_H_
