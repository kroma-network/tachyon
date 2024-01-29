// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_BASE_COLUMN_KEY_STRINGIFIER_H_
#define TACHYON_ZK_PLONK_BASE_COLUMN_KEY_STRINGIFIER_H_

#include <ostream>

#include "tachyon/base/strings/rust_stringifier.h"
#include "tachyon/zk/plonk/base/column_key.h"
#include "tachyon/zk/plonk/base/column_type_stringifier.h"

namespace tachyon::base::internal {

struct AdviceColumnType {
  zk::Phase phase;
  explicit AdviceColumnType(zk::Phase phase) : phase(phase) {}
};

template <>
class RustDebugStringifier<AdviceColumnType> {
 public:
  static std::ostream& AppendToStream(std::ostream& os, RustFormatter& fmt,
                                      const AdviceColumnType& column_type) {
    return os << fmt.DebugStruct("Advice")
                     .Field("phase", column_type.phase)
                     .Finish();
  }
};

template <zk::ColumnType C>
class RustDebugStringifier<zk::ColumnKey<C>> {
 public:
  static std::ostream& AppendToStream(std::ostream& os, RustFormatter& fmt,
                                      const zk::ColumnKey<C>& column) {
    // NOTE(chokobole): See
    // https://github.com/kroma-network/halo2/blob/7d0a36990452c8e7ebd600de258420781a9b7917/halo2_proofs/src/plonk/circuit.rs#L26-L31.
    if (column.phase() == zk::kFirstPhase) {
      return os << fmt.DebugStruct("Column")
                       .Field("index", column.index())
                       .Field("column_type", column.type())
                       .Finish();
    } else {
      // NOTE(dongchangYoo): The |ColumnType| in rust-halo2 includes |Phase|.
      // Example of stringify rules:
      // - |ColumnType| with |Phase(0)| => "Advice",
      // - |ColumnType| with the other phases => "Advice { phase: Phase(1) }"
      AdviceColumnType advice_column_type(column.phase());
      return os << fmt.DebugStruct("Column")
                       .Field("index", column.index())
                       .Field("column_type", advice_column_type)
                       .Finish();
    }
  }
};

}  // namespace tachyon::base::internal

#endif  // TACHYON_ZK_PLONK_BASE_COLUMN_KEY_STRINGIFIER_H_
