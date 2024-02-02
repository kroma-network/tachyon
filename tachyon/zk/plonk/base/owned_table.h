// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_BASE_OWNED_TABLE_H_
#define TACHYON_ZK_PLONK_BASE_OWNED_TABLE_H_

#include <utility>
#include <vector>

#include "tachyon/zk/plonk/base/table_base.h"

namespace tachyon::zk::plonk {

template <typename PolyOrEvals>
class OwnedTable : public TableBase<PolyOrEvals> {
 public:
  OwnedTable() = default;
  OwnedTable(std::vector<PolyOrEvals>&& fixed_columns,
             std::vector<PolyOrEvals>&& advice_columns,
             std::vector<PolyOrEvals>&& instance_columns)
      : fixed_columns_(std::move(fixed_columns)),
        advice_columns_(std::move(advice_columns)),
        instance_columns_(std::move(instance_columns)) {}

  // TableBase<PolyOrEvals> methods
  absl::Span<const PolyOrEvals> GetFixedColumns() const override {
    return fixed_columns_;
  }
  absl::Span<const PolyOrEvals> GetAdviceColumns() const override {
    return advice_columns_;
  }
  absl::Span<const PolyOrEvals> GetInstanceColumns() const override {
    return instance_columns_;
  }

 protected:
  std::vector<PolyOrEvals> fixed_columns_;
  std::vector<PolyOrEvals> advice_columns_;
  std::vector<PolyOrEvals> instance_columns_;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_BASE_OWNED_TABLE_H_
