// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_QUERY_H_
#define TACHYON_ZK_PLONK_CIRCUIT_QUERY_H_

#include <string>

#include "absl/strings/substitute.h"

#include "tachyon/zk/plonk/circuit/column.h"
#include "tachyon/zk/plonk/circuit/rotation.h"

namespace tachyon::zk {

template <ColumnType C>
class QueryData {
 public:
  QueryData() = default;
  QueryData(Rotation rotation, const Column<C>& column)
      : rotation_(rotation), column_(column) {}

  Rotation rotation() const { return rotation_; }
  const Column<C>& column() const { return column_; }

  bool operator==(const QueryData& other) const {
    return rotation_ == other.rotation_ && column_ == other.column_;
  }
  bool operator!=(const QueryData& other) const { return !operator==(other); }

  std::string ToString() const {
    return absl::Substitute("{rotation: $0, column: $1}", rotation_.ToString(),
                            column_.ToString());
  }

 protected:
  Rotation rotation_;
  Column<C> column_;
};

using FixedQueryData = QueryData<ColumnType::kFixed>;
using InstanceQueryData = QueryData<ColumnType::kInstance>;
using AdviceQueryData = QueryData<ColumnType::kAdvice>;

template <ColumnType C>
class Query : public QueryData<C> {
 public:
  Query() = default;
  Query(size_t index, Rotation rotation, const Column<C>& column)
      : QueryData<C>(rotation, column), index_(index) {}

  size_t index() const { return index_; }

  bool operator==(const Query& other) const {
    return QueryData<C>::operator==(other) && index_ == other.index_;
  }
  bool operator!=(const Query& other) const { return !operator==(other); }

  std::string ToString() const {
    return absl::Substitute("{index: $0, rotation: $1, column: $2}", index_,
                            this->rotation_.ToString(),
                            this->column_.ToString());
  }

 private:
  size_t index_ = 0;
};

using FixedQuery = Query<ColumnType::kFixed>;
using InstanceQuery = Query<ColumnType::kInstance>;
using AdviceQuery = Query<ColumnType::kAdvice>;

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_QUERY_H_
