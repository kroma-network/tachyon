// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_COLUMN_H_
#define TACHYON_ZK_PLONK_CIRCUIT_COLUMN_H_

#include <stddef.h>

#include <string>
#include <utility>

#include "absl/hash/hash.h"
#include "absl/strings/substitute.h"

#include "tachyon/zk/plonk/circuit/column_type.h"
#include "tachyon/zk/plonk/circuit/phase.h"

namespace tachyon::zk {

// AnyColumn c = FixedColumn(1); -> ok
// AnyColumn c = AnyColumn(1); -> ok
// FixedColumn c = FixedColumn(1); -> ok
// FixedColumn c = InstanceColumn(1); -> no
// FixedColumn c = AnyColumn(1); -> yes
// See column_unittest.cc
template <ColumnType C>
class Column {
 public:
  constexpr static ColumnType kDefaultType = C;

  Column() = default;
  explicit Column(size_t index) : index_(index) {}

  // NOTE(chokobole): AdviceColumn can be constructed with an additional
  // argument |phase|.
  //
  //   AdviceColumn column(1, kSecondPhase);
  template <ColumnType C2 = C,
            std::enable_if_t<C2 == ColumnType::kAdvice>* = nullptr>
  Column(size_t index, Phase phase) : index_(index), phase_(phase) {}

  // NOTE(chokobole): in this case, |type_| is changed!
  //
  //   AnyColumn c(FixedColumn(1));
  //   CHECK_EQ(c.type(), ColumnType::kFixed);
  //
  //   AnyColumn c(AdviceColumn(1, kSecondPhase));
  //   CHECK_EQ(c.type(), ColumnType::kAdvice);
  //   CHECK_EQ(c.phase(), kSecondPhase);
  template <ColumnType C2,
            std::enable_if_t<C == ColumnType::kAny && C != C2>* = nullptr>
  Column(const Column<C2>& other)
      : type_(other.type_), index_(other.index_), phase_(other.phase_) {}

  // NOTE(chokobole): in this case, |type_| is not changed!
  //
  //   FixedColumn c(AnyColumn(1));
  //   CHECK_EQ(c.type(), ColumnType::kFixed);
  //
  //   AnyColumn c(AdviceColumn(1, kSecondPhase));
  //   AdviceColumn c2(c);
  //   CHECK_EQ(c2.type(), ColumnType::kAdvice);
  //   CHECK_EQ(c2.phase(), kSecondPhase);
  template <ColumnType C2, std::enable_if_t<C != ColumnType::kAny &&
                                            C2 == ColumnType::kAny>* = nullptr>
  Column(const Column<C2>& other)
      : index_(other.index_), phase_(other.phase_) {}

  // FixedColumn c(FixedColumn(1));
  Column(const Column& other) = default;

  // NOTE(chokobole): in this case, |type_| is changed!
  //
  //   AnyColumn c;
  //   c = FixedColumn(1);
  //   CHECK_EQ(c.type(), ColumnType::kFixed);
  //
  //   AnyColumn c;
  //   c = AdviceColumn(1, kSecondPhase);
  //   CHECK_EQ(c.type(), ColumnType::kAdvice);
  //   CHECK_EQ(c.phase(), kSecondPhase);
  template <ColumnType C2,
            std::enable_if_t<C == ColumnType::kAny && C != C2>* = nullptr>
  Column& operator=(const Column<C2>& other) {
    type_ = other.type_;
    index_ = other.index_;
    phase_ = other.phase_;
    return *this;
  }

  // NOTE(chokobole): in this case, |type_| is not changed!
  //
  //   FixedColumn c;
  //   c = AnyColumn(1);
  //   CHECK_EQ(c.type(), ColumnType::kFixed);
  //
  //   AdviceColumn c;
  //   c = AnyColumn(AdviceColumn(1, kSecondPhase));
  //   CHECK_EQ(c.type(), ColumnType::kAdvice);
  //   CHECK_EQ(c.phase(), kSecondPhase);
  template <ColumnType C2, std::enable_if_t<C != ColumnType::kAny &&
                                            C2 == ColumnType::kAny>* = nullptr>
  Column& operator=(const Column<C2>& other) {
    index_ = other.index_;
    phase_ = other.phase_;
    return *this;
  }

  // FixedColumn c;
  // c = FixedColumn(1);
  Column& operator=(const Column& other) = default;

  ColumnType type() const { return type_; }
  size_t index() const { return index_; }
  Phase phase() const { return phase_; }

  bool operator==(const Column& other) const {
    if (type_ == ColumnType::kAdvice) {
      return type_ == other.type_ && index_ == other.index_ &&
             phase_ == other.phase_;
    } else {
      return type_ == other.type_ && index_ == other.index_;
    }
  }
  bool operator!=(const Column& other) const { return !operator==(other); }

  std::string ToString() const {
    if (type_ == ColumnType::kAdvice) {
      return absl::Substitute("{type: $0, index: $1, phase: $2}",
                              ColumnTypeToString(type_), index_,
                              phase_.ToString());
    } else {
      return absl::Substitute("{type: $0, index: $1}",
                              ColumnTypeToString(type_), index_);
    }
  }

 private:
  template <typename H>
  friend H AbslHashValue(H h, const Column<C>& column);

  template <ColumnType C2>
  friend class Column;

  ColumnType type_ = C;
  size_t index_ = 0;
  Phase phase_;
};

using AnyColumn = Column<ColumnType::kAny>;
using FixedColumn = Column<ColumnType::kFixed>;
using AdviceColumn = Column<ColumnType::kAdvice>;
using InstanceColumn = Column<ColumnType::kInstance>;

template <typename H, ColumnType C>
H AbslHashValue(H h, const Column<C>& column) {
  return H::combine(std::move(h), column.type_, column.index_);
}

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_COLUMN_H_
