// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_TABLE_H_
#define TACHYON_ZK_PLONK_CIRCUIT_TABLE_H_

#include <string>

#include "tachyon/math/base/rational_field.h"
#include "tachyon/zk/plonk/circuit/table_column.h"
#include "tachyon/zk/plonk/error.h"
#include "tachyon/zk/value.h"

namespace tachyon::zk {

// A lookup table in the circuit.
template <typename F>
class Table {
 public:
  class Layouter {
   public:
    using AssignCallback = base::OnceCallback<Value<math::RationalField<F>>()>;

    virtual ~Layouter() = default;

    // Assign a fixed value to a table cell.
    //
    // Return an error if the table cell has already been assigned.
    virtual Error AssignCell(std::string_view name, const TableColumn& column,
                             size_t offset, AssignCallback assign) {
      return Error::kNone;
    }
  };

  explicit Table(Layouter* layouter) : layouter_(layouter) {}

 private:
  Layouter* const layouter_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_TABLE_H_
