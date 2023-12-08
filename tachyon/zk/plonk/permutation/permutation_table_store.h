// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_TABLE_STORE_H_
#define TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_TABLE_STORE_H_

#include <vector>

#include "absl/types/span.h"
#include "gtest/gtest_prod.h"

#include "tachyon/zk/plonk/circuit/table.h"
#include "tachyon/zk/plonk/permutation/permuted_table.h"
#include "tachyon/zk/plonk/permutation/unpermuted_table.h"

namespace tachyon::zk {

template <typename Evals>
class PermutationTableStore {
 public:
  using F = typename Evals::Field;

  PermutationTableStore(const std::vector<AnyColumnKey>& column_keys,
                        const Table<Evals>& table,
                        const PermutedTable<Evals>& permuted_table,
                        const UnpermutedTable<Evals>& unpermuted_table,
                        size_t chunk_size)
      : column_keys_(column_keys),
        table_(table),
        permuted_table_(permuted_table),
        unpermuted_table_(unpermuted_table),
        chunk_size_(chunk_size) {}

  size_t GetChunkSize(size_t chunk_idx) const {
    size_t total_size = column_keys_.size();
    size_t chunk_num = GetChunkNum();
    CHECK_LT(chunk_idx, chunk_num);

    if (chunk_idx < chunk_num - 1) {
      return chunk_size_;
    } else {
      return total_size - (chunk_num - 1) * chunk_size_;
    }
  }

  std::vector<base::Ref<const Evals>> GetValueColumns(size_t chunk_idx) const {
    size_t chunk_offset = GetChunkOffset(chunk_idx);
    size_t chunk_size = GetChunkSize(chunk_idx);
    absl::Span<const AnyColumnKey> keys =
        absl::MakeConstSpan(column_keys_.data() + chunk_offset, chunk_size);
    return table_.GetColumns(keys);
  }

  std::vector<base::Ref<const Evals>> GetPermutedColumns(
      size_t chunk_idx) const {
    return GetColumns(permuted_table_, chunk_idx);
  }

  std::vector<base::Ref<const Evals>> GetUnpermutedColumns(
      size_t chunk_idx) const {
    return GetColumns(unpermuted_table_, chunk_idx);
  }

 private:
  FRIEND_TEST(PermutationTableStoreTest, GetColumns);

  size_t GetChunkNum() const {
    return (column_keys_.size() + chunk_size_ - 1) / chunk_size_;
  }

  size_t GetChunkOffset(size_t chunk_idx) const {
    return chunk_idx * chunk_size_;
  }

  template <typename T>
  auto GetColumns(const T& table, size_t chunk_idx) const {
    size_t chunk_offset = GetChunkOffset(chunk_idx);
    size_t chunk_size = GetChunkSize(chunk_idx);
    return table.GetColumns(
        base::Range<size_t>(chunk_offset, chunk_offset + chunk_size));
  }

  const std::vector<AnyColumnKey>& column_keys_;
  const Table<Evals>& table_;
  const PermutedTable<Evals>& permuted_table_;
  const UnpermutedTable<Evals>& unpermuted_table_;
  // The number of element in a chunk.
  size_t chunk_size_ = 0;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_TABLE_STORE_H_
