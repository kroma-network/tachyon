// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_QUERY_H_
#define TACHYON_ZK_PLONK_CIRCUIT_QUERY_H_

#include <string>

#include "absl/strings/substitute.h"

#include "tachyon/zk/plonk/circuit/column_type.h"
#include "tachyon/zk/plonk/circuit/phase.h"
#include "tachyon/zk/plonk/circuit/rotation.h"

namespace tachyon::zk {

template <ColumnType C>
class Query {
 public:
  constexpr static ColumnType kType = C;

  Query() = default;
  Query(size_t index, size_t column_index, Rotation rotation)
      : index_(index), column_index_(column_index), rotation_(rotation) {}

  size_t index() const { return index_; }
  size_t column_index() const { return column_index_; }
  Rotation rotation() const { return rotation_; }

  std::string ToString() const {
    return absl::Substitute(
        "{type: $0, index: $1, column_index: $2, rotation: $3}",
        ColumnTypeToString(C), index_, column_index_, rotation_.ToString());
  }

 protected:
  size_t index_ = 0;
  size_t column_index_ = 0;
  Rotation rotation_;
};

using FixedQuery = Query<ColumnType::kFixed>;
using InstanceQuery = Query<ColumnType::kInstance>;

class TACHYON_EXPORT AdviceQuery : public Query<ColumnType::kAdvice> {
 public:
  AdviceQuery() = default;
  AdviceQuery(size_t index, size_t column_index, Rotation rotation, Phase phase)
      : Query<ColumnType::kAdvice>(index, column_index, rotation),
        phase_(phase) {}

  Phase phase() const { return phase_; }

  std::string ToString() const {
    return absl::Substitute(
        "{type: $0, index: $1, column_index: $2, rotation: $3, phase: $4}",
        ColumnTypeToString(ColumnType::kAdvice), index_, column_index_,
        rotation_.ToString(), phase_.ToString());
  }

 private:
  Phase phase_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_QUERY_H_
