// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_HALO2_STRINGIFIERS_PERMUTATION_ARGUMENT_STRINGIFIER_H_
#define TACHYON_ZK_PLONK_HALO2_STRINGIFIERS_PERMUTATION_ARGUMENT_STRINGIFIER_H_

#include <ostream>

#include "tachyon/base/strings/rust_stringifier.h"
#include "tachyon/zk/plonk/halo2/stringifiers/column_key_stringifier.h"
#include "tachyon/zk/plonk/permutation/permutation_argument.h"

namespace tachyon::base::internal {

template <>
class RustDebugStringifier<zk::plonk::PermutationArgument> {
 public:
  static std::ostream& AppendToStream(
      std::ostream& os, RustFormatter& fmt,
      const zk::plonk::PermutationArgument& argument) {
    return os << fmt.DebugStruct("Argument")
                     .Field("columns", argument.columns())
                     .Finish();
  }
};

}  // namespace tachyon::base::internal

#endif  // TACHYON_ZK_PLONK_HALO2_STRINGIFIERS_PERMUTATION_ARGUMENT_STRINGIFIER_H_
