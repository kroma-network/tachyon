#ifndef TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_SIMPLE_LOOKUP_TABLE_LAYOUTER_H_
#define TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_SIMPLE_LOOKUP_TABLE_LAYOUTER_H_

#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"

#include "tachyon/base/containers/contains.h"
#include "tachyon/math/base/rational_field.h"
#include "tachyon/zk/base/value.h"
#include "tachyon/zk/plonk/circuit/assignment.h"
#include "tachyon/zk/plonk/circuit/lookup_table.h"
#include "tachyon/zk/plonk/circuit/lookup_table_column.h"

namespace tachyon::zk {

template <typename F>
class SimpleLookupTableLayouter : public LookupTable<F>::Layouter {
 public:
  using AssignCallback = typename LookupTable<F>::Layouter::AssignCallback;

  struct Value {
    // The default value to fill a table column with.
    //
    // - The outer |std::optional| tracks whether the value in row 0 of the
    //   table column has been assigned yet. This will always have value once a
    //   valid table has been completely assigned.
    // - The inner |zk::Value| tracks whether the underlying
    //   |math::RationalField| is evaluating witnesses or not.
    using DefaultValue = std::optional<zk::Value<math::RationalField<F>>>;

    DefaultValue default_value;
    std::vector<bool> assigned;
  };

  SimpleLookupTableLayouter(Assignment<F>* assignment,
                            const std::vector<LookupTableColumn>* used_columns)
      : assignment_(assignment), used_columns_(used_columns) {}

  const absl::flat_hash_map<LookupTableColumn, Value>& values() const {
    return values_;
  }

  // Table<F>::Layouter methods
  Error AssignCell(std::string_view name, const LookupTableColumn& column,
                   size_t offset, AssignCallback assign) override {
    if (base::Contains(*used_columns_, column)) {
      return Error::kSynthesis;
    }

    zk::Value<math::RationalField<F>> value =
        zk::Value<math::RationalField<F>>::Unknown();
    assignment_->AssignFixed(
        name, column.column(), offset, [&value, assign = std::move(assign)]() {
          zk::Value<math::RationalField<F>> ret = std::move(assign).Run();
          value = ret;
          return ret;
        });

    if (offset == 0) {
      if (value.default_value.has_value()) {
        // Use the value at offset 0 as the default value for this table column.
        value.default_value = value;
      } else {
        // Since there is already an existing default value for this table
        // column, the caller should not be attempting to assign another
        // value at offset 0.
        return Error::kSynthesis;
      }
    }
    if (value.assigned.size() <= offset) {
      value.assigned.resize(offset + 1, false);
    }
    value.assigned[offset] = true;
    return Error::kNone;
  }

 private:
  Assignment<F>* const assignment_;
  const std::vector<LookupTableColumn>* const used_columns_;
  absl::flat_hash_map<LookupTableColumn, Value> values_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_SIMPLE_LOOKUP_TABLE_LAYOUTER_H_
