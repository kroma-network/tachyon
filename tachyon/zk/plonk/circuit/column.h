// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_COLUMN_H_
#define TACHYON_ZK_PLONK_CIRCUIT_COLUMN_H_

#include <stddef.h>

#include <string>

#include "absl/strings/substitute.h"

#include "tachyon/zk/plonk/circuit/column_type.h"

namespace tachyon::zk {

// AnyColumn c = FixedColumn(1); -> ok
// AnyColumn c = AnyColumn(1); -> ok
// FixedColumn c = FixedColumn(1); -> ok
// FixedColumn c = InstanceColumn(1); -> no
// FixedColumn c = AnyColumn(1); -> no
// See column_unittest.cc
template <ColumnType C>
class Column {
 public:
  constexpr static ColumnType kDefaultType = C;

  Column() = default;
  explicit Column(size_t index) : index_(index) {}

  template <ColumnType C2,
            std::enable_if_t<C == ColumnType::kAny && C != C2>* = nullptr>
  Column(const Column<C2>& other) : type_(other.type_), index_(other.index_) {}
  Column(const Column& other) = default;
  template <ColumnType C2,
            std::enable_if_t<C == ColumnType::kAny && C != C2>* = nullptr>
  Column& operator=(const Column<C2>& other) {
    type_ = other.type_;
    index_ = other.index_;
    return *this;
  }
  Column& operator=(const Column& other) = default;

  ColumnType type() const { return type_; }
  size_t index() const { return index_; }

  bool operator==(const Column& other) const {
    return type_ == other.type_ && index_ == other.index_;
  }
  bool operator!=(const Column& other) const { return !operator==(other); }

  std::string ToString() const {
    return absl::Substitute("{type: $0, index: $1}", ColumnTypeToString(type_),
                            index_);
  }

 private:
  template <ColumnType C2>
  friend class Column;

  ColumnType type_ = C;
  size_t index_ = 0;
};

using AnyColumn = Column<ColumnType::kAny>;
using FixedColumn = Column<ColumnType::kFixed>;
using AdviceColumn = Column<ColumnType::kAdvice>;
using InstanceColumn = Column<ColumnType::kInstance>;

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_COLUMN_H_
