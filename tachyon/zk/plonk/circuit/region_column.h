// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_REGION_COLUMN_H_
#define TACHYON_ZK_PLONK_CIRCUIT_REGION_COLUMN_H_

#include <string>
#include <utility>

#include "absl/hash/hash.h"

#include "tachyon/export.h"
#include "tachyon/zk/plonk/circuit/column_key.h"
#include "tachyon/zk/plonk/circuit/selector.h"

namespace tachyon::zk {

class TACHYON_EXPORT RegionColumn {
 public:
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
  if (m.type() == RegionColumn::Type::kColumn) {
    return H::combine(std::move(h), m.column());
  } else {
    return H::combine(std::move(h), m.selector());
  }
}

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_REGION_COLUMN_H_
