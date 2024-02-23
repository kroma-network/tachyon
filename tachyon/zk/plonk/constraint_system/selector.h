// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_SELECTOR_H_
#define TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_SELECTOR_H_

#include <stddef.h>

#include <string>
#include <utility>

#include "absl/hash/hash.h"
#include "absl/strings/substitute.h"

#include "tachyon/export.h"
#include "tachyon/zk/base/row_types.h"

namespace tachyon::zk::plonk {

template <typename F>
class Region;

// NOTE(dongchangYoo): Selector class is copyable, assignable, and occupy 65
// bits per instance. Prefer to pass them by value.
class TACHYON_EXPORT Selector {
 public:
  static Selector Simple(size_t index) { return {index, true}; }

  static Selector Complex(size_t index) { return {index, false}; }

  size_t index() const { return index_; }
  bool is_simple() const { return is_simple_; }

  bool operator==(Selector other) const {
    return index_ == other.index_ && is_simple_ == other.is_simple_;
  }
  bool operator!=(Selector other) const { return !operator==(other); }

  // Defined in region.h
  template <typename F>
  void Enable(Region<F>& region, RowIndex offset) const;

  std::string ToString() const {
    return absl::Substitute("{index: $0, is_simple: $1}", index_, is_simple_);
  }

 private:
  Selector(size_t index, bool is_simple)
      : index_(index), is_simple_(is_simple) {}

  size_t index_ = 0;
  bool is_simple_ = false;
};

template <typename H>
H AbslHashValue(H h, Selector selector) {
  return H::combine(std::move(h), selector.index(), selector.is_simple());
}

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_SELECTOR_H_
