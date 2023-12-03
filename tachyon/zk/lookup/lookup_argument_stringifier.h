// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_LOOKUP_LOOKUP_ARGUMENT_STRINGIFIER_H_
#define TACHYON_ZK_LOOKUP_LOOKUP_ARGUMENT_STRINGIFIER_H_

#include <ostream>

#include "tachyon/base/strings/rust_stringifier.h"
#include "tachyon/zk/expressions/expression_stringifier.h"
#include "tachyon/zk/lookup/lookup_argument.h"

namespace tachyon::base::internal {

template <typename F>
class RustDebugStringifier<zk::LookupArgument<F>> {
 public:
  static std::ostream& AppendToStream(std::ostream& os, RustFormatter& fmt,
                                      const zk::LookupArgument<F>& argument) {
    return os << fmt.DebugStruct("Argument")
                     .Field("input_expressions", argument.input_expressions())
                     .Field("table_expressions", argument.table_expressions())
                     .Finish();
  }
};

}  // namespace tachyon::base::internal

#endif  // TACHYON_ZK_LOOKUP_LOOKUP_ARGUMENT_STRINGIFIER_H_
