// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_COLUMN_STRINGIFIER_H_
#define TACHYON_ZK_PLONK_CIRCUIT_COLUMN_STRINGIFIER_H_

#include <ostream>

#include "tachyon/base/strings/rust_stringifier.h"
#include "tachyon/zk/plonk/circuit/column.h"
#include "tachyon/zk/plonk/circuit/column_type_stringifier.h"

namespace tachyon::base::internal {

template <zk::ColumnType C>
class RustDebugStringifier<zk::Column<C>> {
 public:
  static std::ostream& AppendToStream(std::ostream& os, RustFormatter& fmt,
                                      const zk::Column<C>& column) {
    // NOTE(chokobole): See
    // https://github.com/kroma-network/halo2/blob/7d0a36990452c8e7ebd600de258420781a9b7917/halo2_proofs/src/plonk/circuit.rs#L26-L31.
    return os << fmt.DebugStruct("Column")
                     .Field("index", column.index())
                     .Field("column_type", column.type())
                     .Finish();
  }
};

}  // namespace tachyon::base::internal

#endif  // TACHYON_ZK_PLONK_CIRCUIT_COLUMN_STRINGIFIER_H_
