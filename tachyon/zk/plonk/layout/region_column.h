// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_LAYOUT_REGION_COLUMN_H_
#define TACHYON_ZK_PLONK_LAYOUT_REGION_COLUMN_H_

#include <string>
#include <utility>

#include "absl/hash/hash.h"

#include "tachyon/export.h"
#include "tachyon/zk/plonk/base/column_key.h"
#include "tachyon/zk/plonk/constraint_system/selector.h"

namespace tachyon::zk {

class TACHYON_EXPORT RegionColumn {
 public:
  // NOTE(TomTaehoonKim): THE ORDER OF ELEMENTS ARE IMPORTANT!! DO NOT CHANGE!
  enum class Type {
    kColumn,
    kSelector,
  };

  RegionColumn() {}
  explicit RegionColumn(const AnyColumnKey& column)
      : type_(Type::kColumn), column_(column) {}
  explicit RegionColumn(const Selector& selector)
      : type_(Type::kSelector), selector_(selector) {}

  Type type() const { return type_; }
  const AnyColumnKey& column() const { return column_; }
  const Selector& selector() const { return selector_; }

  bool operator==(const RegionColumn& other) const {
    if (type_ != other.type_) return false;
    if (type_ == Type::kColumn) return column_ == other.column_;
    return selector_ == other.selector_;
  }
  bool operator!=(const RegionColumn& other) const {
    return !operator==(other);
  }

  bool operator<(const RegionColumn& other) const {
    if (type_ == Type::kColumn && other.type_ == Type::kColumn) {
      return column_ < other.column_;
    } else if (type_ == Type::kSelector && other.type_ == Type::kSelector) {
      return selector_.index() < other.selector_.index();
    }
    return static_cast<int>(type_) < static_cast<int>(other.type_);
  }

  std::string ToString() const {
    if (type_ == Type::kColumn) {
      return column_.ToString();
    } else {
      return selector_.ToString();
    }
  }

 private:
  Type type_;
  union {
    AnyColumnKey column_;
    Selector selector_;
  };
};

template <typename H>
H AbslHashValue(H h, const RegionColumn& m) {
  // NOTE(chokobole): |kTypePrefixAdder| should prevent it from being a suffix
  // of the other.
  // See https://abseil.io/docs/cpp/guides/hash#the-abslhashvalue-overload
  constexpr static int kTypePrefixAdder = 100;
  if (m.type() == RegionColumn::Type::kColumn) {
    return H::combine(std::move(h),
                      kTypePrefixAdder + static_cast<int>(m.type()),
                      m.column());
  } else {
    return H::combine(std::move(h),
                      kTypePrefixAdder + static_cast<int>(m.type()),
                      m.selector());
  }
}

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_LAYOUT_REGION_COLUMN_H_
