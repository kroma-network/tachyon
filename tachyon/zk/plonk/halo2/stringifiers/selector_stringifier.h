// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_HALO2_STRINGIFIERS_SELECTOR_STRINGIFIER_H_
#define TACHYON_ZK_PLONK_HALO2_STRINGIFIERS_SELECTOR_STRINGIFIER_H_

#include <ostream>

#include "tachyon/base/strings/rust_stringifier.h"
#include "tachyon/zk/plonk/constraint_system/selector.h"

namespace tachyon::base::internal {

template <>
class RustDebugStringifier<zk::plonk::Selector> {
 public:
  static std::ostream& AppendToStream(std::ostream& os, RustFormatter& fmt,
                                      zk::plonk::Selector selector) {
    return os << fmt.DebugTuple("Selector")
                     .Field(selector.index())
                     .Field(selector.is_simple())
                     .Finish();
  }
};

}  // namespace tachyon::base::internal

#endif  // TACHYON_ZK_PLONK_HALO2_STRINGIFIERS_SELECTOR_STRINGIFIER_H_
