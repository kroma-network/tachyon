// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_REF_TABLE_H_
#define TACHYON_ZK_PLONK_CIRCUIT_REF_TABLE_H_

#include <vector>

#include "tachyon/zk/plonk/circuit/table_base.h"

namespace tachyon::zk {

template <typename PolyOrEvals>
class RefTable : public TableBase<PolyOrEvals> {
 public:
  RefTable() = default;
  RefTable(absl::Span<const PolyOrEvals> fixed_columns,
           absl::Span<const PolyOrEvals> advice_columns,
           absl::Span<const PolyOrEvals> instance_columns)
      : fixed_columns_(fixed_columns),
        advice_columns_(advice_columns),
        instance_columns_(instance_columns) {}

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
  absl::Span<const PolyOrEvals> fixed_columns_;
  absl::Span<const PolyOrEvals> advice_columns_;
  absl::Span<const PolyOrEvals> instance_columns_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_REF_TABLE_H_
