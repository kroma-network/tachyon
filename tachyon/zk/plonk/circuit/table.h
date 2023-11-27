// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_TABLE_H_
#define TACHYON_ZK_PLONK_CIRCUIT_TABLE_H_

#include <vector>

#include "absl/types/span.h"

#include "tachyon/base/logging.h"
#include "tachyon/zk/base/ref.h"
#include "tachyon/zk/plonk/circuit/column_key.h"

namespace tachyon::zk {

template <typename Evals>
class Table {
 public:
  Table() = default;
  Table(absl::Span<const Evals> fixed_columns,
        absl::Span<const Evals> advice_columns,
        absl::Span<const Evals> instance_columns)
      : fixed_columns_(fixed_columns),
        advice_columns_(advice_columns),
        instance_columns_(instance_columns) {}

  absl::Span<const Evals> fixed_columns() const { return fixed_columns_; }
  absl::Span<const Evals> advice_columns() const { return advice_columns_; }
  absl::Span<const Evals> instance_columns() const { return instance_columns_; }

  Ref<const Evals> GetColumn(const ColumnKeyBase& column_key) const {
    switch (column_key.type()) {
      case ColumnType::kFixed:
        return Ref<const Evals>(&fixed_columns_[column_key.index()]);
      case ColumnType::kAdvice:
        return Ref<const Evals>(&advice_columns_[column_key.index()]);
      case ColumnType::kInstance:
        return Ref<const Evals>(&instance_columns_[column_key.index()]);
      case ColumnType::kAny:
        break;
    }
    NOTREACHED();
    return Ref<const Evals>();
  }

  std::vector<Ref<const Evals>> GetColumns(
      const std::vector<ColumnKeyBase>& column_keys) const {
    std::vector<Ref<const Evals>> ret;
    ret.reserve(column_keys.size());
    for (const ColumnKeyBase& column_key : column_keys) {
      ret.push_back(GetColumn(column_key));
    }
    return ret;
  }

 protected:
  absl::Span<const Evals> fixed_columns_;
  absl::Span<const Evals> advice_columns_;
  absl::Span<const Evals> instance_columns_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_TABLE_H_
