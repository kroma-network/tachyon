// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_LOOKUP_TABLE_H_
#define TACHYON_ZK_PLONK_CIRCUIT_LOOKUP_TABLE_H_

#include <string>

#include "tachyon/math/base/rational_field.h"
#include "tachyon/zk/base/value.h"
#include "tachyon/zk/plonk/circuit/lookup_table_column.h"
#include "tachyon/zk/plonk/error.h"

namespace tachyon::zk {

// A lookup table in the circuit.
template <typename F>
class LookupTable {
 public:
  class Layouter {
   public:
    using AssignCallback = base::OnceCallback<Value<math::RationalField<F>>()>;

    virtual ~Layouter() = default;

    // Assign a fixed value to a table cell.
    //
    // Return an error if the table cell has already been assigned.
    virtual Error AssignCell(std::string_view name,
                             const LookupTableColumn& column, size_t offset,
                             AssignCallback assign) {
      return Error::kNone;
    }
  };

  explicit LookupTable(Layouter* layouter) : layouter_(layouter) {}

 private:
  Layouter* const layouter_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_LOOKUP_TABLE_H_
