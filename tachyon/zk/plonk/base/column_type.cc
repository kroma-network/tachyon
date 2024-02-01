// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/base/column_type.h"

#include "tachyon/base/logging.h"

namespace tachyon::zk::plonk {

std::string_view ColumnTypeToString(ColumnType type) {
  switch (type) {
    case ColumnType::kAny:
      return "Any";
    case ColumnType::kFixed:
      return "Fixed";
    case ColumnType::kAdvice:
      return "Advice";
    case ColumnType::kInstance:
      return "Instance";
  }
  NOTREACHED();
  return "";
}

}  // namespace tachyon::zk::plonk
