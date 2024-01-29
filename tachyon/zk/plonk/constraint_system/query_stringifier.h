// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_QUERY_STRINGIFIER_H_
#define TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_QUERY_STRINGIFIER_H_

#include <ostream>

#include "tachyon/base/strings/rust_stringifier.h"
#include "tachyon/zk/plonk/circuit/column_key_stringifier.h"
#include "tachyon/zk/plonk/constraint_system/query.h"
#include "tachyon/zk/plonk/constraint_system/rotation_stringifier.h"

namespace tachyon::base::internal {

template <zk::ColumnType C>
class RustDebugStringifier<zk::QueryData<C>> {
 public:
  static std::ostream& AppendToStream(std::ostream& os, RustFormatter& fmt,
                                      const zk::QueryData<C>& query) {
    // NOTE(chokobole): See
    // https://github.com/kroma-network/halo2/blob/7d0a36990452c8e7ebd600de258420781a9b7917/halo2_proofs/src/plonk/circuit.rs#L1382.
    return os << fmt.DebugTuple("")
                     .Field(query.column())
                     .Field(query.rotation())
                     .Finish();
  }
};

}  // namespace tachyon::base::internal

#endif  // TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_QUERY_STRINGIFIER_H_
