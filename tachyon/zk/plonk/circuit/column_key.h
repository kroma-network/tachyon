// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_COLUMN_KEY_H_
#define TACHYON_ZK_PLONK_CIRCUIT_COLUMN_KEY_H_

#include <stddef.h>

#include <string>
#include <utility>

#include "absl/hash/hash.h"
#include "absl/strings/substitute.h"

#include "tachyon/zk/plonk/circuit/column_type.h"
#include "tachyon/zk/plonk/circuit/phase.h"

namespace tachyon::zk {

class TACHYON_EXPORT ColumnKeyBase {
 public:
  ColumnKeyBase() = default;
  ColumnKeyBase(ColumnType type, size_t index) : type_(type), index_(index) {}

  ColumnType type() const { return type_; }
  size_t index() const { return index_; }

  bool operator==(const ColumnKeyBase& other) const {
    return type_ == other.type_ && index_ == other.index_;
  }
  bool operator!=(const ColumnKeyBase& other) const {
    return !operator==(other);
  }

  std::string ToString() const {
    return absl::Substitute("{type: $0, index: $1}", ColumnTypeToString(type_),
                            index_);
  }

 protected:
  ColumnType type_ = ColumnType::kAny;
  size_t index_ = 0;
};

template <typename H>
H AbslHashValue(H h, const ColumnKeyBase& column) {
  return H::combine(std::move(h), column.type(), column.index());
}

// AnyColumnKey c = FixedColumnKey(1); -> ok
// AnyColumnKey c = AnyColumnKey(1); -> ok
// FixedColumnKey c = FixedColumnKey(1); -> ok
// FixedColumnKey c = InstanceColumnKey(1); -> no
// FixedColumnKey c = AnyColumnKey(1); -> yes
// See column_unittest.cc
template <ColumnType C>
class ColumnKey : public ColumnKeyBase {
 public:
  constexpr static ColumnType kDefaultType = C;

  ColumnKey() : ColumnKeyBase(C, 0) {}
  explicit ColumnKey(size_t index) : ColumnKeyBase(C, index) {}

  // NOTE(chokobole): AdviceColumnKey can be constructed with an additional
  // argument |phase|.
  //
  //   AdviceColumnKey column(1, kSecondPhase);
  template <ColumnType C2 = C,
            std::enable_if_t<C2 == ColumnType::kAdvice>* = nullptr>
  ColumnKey(size_t index, Phase phase)
      : ColumnKeyBase(C, index), phase_(phase) {}

  // NOTE(chokobole): in this case, |type_| is changed!
  //
  //   AnyColumnKey c(FixedColumnKey(1));
  //   CHECK_EQ(c.type(), ColumnType::kFixed);
  //
  //   AnyColumnKey c(AdviceColumnKey(1, kSecondPhase));
  //   CHECK_EQ(c.type(), ColumnType::kAdvice);
  //   CHECK_EQ(c.phase(), kSecondPhase);
  template <ColumnType C2,
            std::enable_if_t<C == ColumnType::kAny && C != C2>* = nullptr>
  ColumnKey(const ColumnKey<C2>& other)
      : ColumnKeyBase(other.type_, other.index_), phase_(other.phase_) {}

  // NOTE(chokobole): in this case, |type_| is not changed!
  //
  //   FixedColumnKey c(AnyColumnKey(1));
  //   CHECK_EQ(c.type(), ColumnType::kFixed);
  //
  //   AnyColumnKey c(AdviceColumnKey(1, kSecondPhase));
  //   AdviceColumnKey c2(c);
  //   CHECK_EQ(c2.type(), ColumnType::kAdvice);
  //   CHECK_EQ(c2.phase(), kSecondPhase);
  template <ColumnType C2, std::enable_if_t<C != ColumnType::kAny &&
                                            C2 == ColumnType::kAny>* = nullptr>
  ColumnKey(const ColumnKey<C2>& other)
      : ColumnKeyBase(C, other.index_), phase_(other.phase_) {}

  // FixedColumnKey c(FixedColumnKey(1));
  ColumnKey(const ColumnKey& other) = default;

  // NOTE(chokobole): in this case, |type_| is changed!
  //
  //   AnyColumnKey c;
  //   c = FixedColumnKey(1);
  //   CHECK_EQ(c.type(), ColumnType::kFixed);
  //
  //   AnyColumnKey c;
  //   c = AdviceColumnKey(1, kSecondPhase);
  //   CHECK_EQ(c.type(), ColumnType::kAdvice);
  //   CHECK_EQ(c.phase(), kSecondPhase);
  template <ColumnType C2,
            std::enable_if_t<C == ColumnType::kAny && C != C2>* = nullptr>
  ColumnKey& operator=(const ColumnKey<C2>& other) {
    ColumnKeyBase::operator=(other);
    phase_ = other.phase_;
    return *this;
  }

  // NOTE(chokobole): in this case, |type_| is not changed!
  //
  //   FixedColumnKey c;
  //   c = AnyColumnKey(1);
  //   CHECK_EQ(c.type(), ColumnType::kFixed);
  //
  //   AdviceColumnKey c;
  //   c = AnyColumnKey(AdviceColumnKey(1, kSecondPhase));
  //   CHECK_EQ(c.type(), ColumnType::kAdvice);
  //   CHECK_EQ(c.phase(), kSecondPhase);
  template <ColumnType C2, std::enable_if_t<C != ColumnType::kAny &&
                                            C2 == ColumnType::kAny>* = nullptr>
  ColumnKey& operator=(const ColumnKey<C2>& other) {
    index_ = other.index_;
    phase_ = other.phase_;
    return *this;
  }

  // FixedColumnKey c;
  // c = FixedColumnKey(1);
  ColumnKey& operator=(const ColumnKey& other) = default;

  ColumnType type() const { return type_; }
  size_t index() const { return index_; }
  Phase phase() const { return phase_; }

  bool operator==(const ColumnKey& other) const {
    if (!ColumnKeyBase::operator==(other)) return false;
    if (type_ == ColumnType::kAdvice) {
      return phase_ == other.phase_;
    }
    return true;
  }
  bool operator!=(const ColumnKey& other) const { return !operator==(other); }

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
  template <ColumnType C2>
  friend class ColumnKey;

  Phase phase_;
};

using AnyColumnKey = ColumnKey<ColumnType::kAny>;
using FixedColumnKey = ColumnKey<ColumnType::kFixed>;
using AdviceColumnKey = ColumnKey<ColumnType::kAdvice>;
using InstanceColumnKey = ColumnKey<ColumnType::kInstance>;

template <typename H, ColumnType C>
H AbslHashValue(H h, const ColumnKey<C>& column) {
  return H::combine(std::move(h), column.type(), column.index());
}

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_COLUMN_KEY_H_
