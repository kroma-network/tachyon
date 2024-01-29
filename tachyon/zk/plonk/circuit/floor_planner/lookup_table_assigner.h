#ifndef TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_LOOKUP_TABLE_ASSIGNER_H_
#define TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_LOOKUP_TABLE_ASSIGNER_H_

#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"

#include "tachyon/base/functional/identity.h"
#include "tachyon/base/logging.h"
#include "tachyon/zk/plonk/circuit/assignment.h"
#include "tachyon/zk/plonk/circuit/floor_planner/scoped_region.h"
#include "tachyon/zk/plonk/circuit/floor_planner/simple_lookup_table_layouter.h"
#include "tachyon/zk/plonk/circuit/layouter.h"
#include "tachyon/zk/plonk/circuit/lookup_table.h"
#include "tachyon/zk/plonk/circuit/lookup_table_column.h"

namespace tachyon::zk {

template <typename F>
class LookupTableAssigner {
 public:
  using AssignLookupTableCallback =
      typename Layouter<F>::AssignLookupTableCallback;

  LookupTableAssigner(Assignment<F>* assignment,
                      std::vector<LookupTableColumn>& lookup_table_columns)
      : assignment_(assignment), lookup_table_columns_(lookup_table_columns) {}

  void AssignLookupTable(std::string_view name,
                         AssignLookupTableCallback assign) {
    // Assign table cells.
    SimpleLookupTableLayouter<F> lookup_table_layouter(assignment_,
                                                       &lookup_table_columns_);
    {
      ScopedRegion<F> scoped_region(assignment_, name);
      LookupTable<F> table(&lookup_table_layouter);
      std::move(assign).Run(table);
    }
    const absl::flat_hash_map<LookupTableColumn,
                              typename SimpleLookupTableLayouter<F>::Value>&
        values = lookup_table_layouter.values();
    // Check that all table columns have the same length |first_unused|,
    // and all cells up to that length are assigned.
    std::optional<RowIndex> first_unused;
    std::vector<std::optional<RowIndex>> assigned_sizes = base::Map(
        values.begin(), values.end(),
        [](const std::pair<LookupTableColumn,
                           typename SimpleLookupTableLayouter<F>::Value>&
               entry) {
          const auto& [column, value] = entry;
          if (std::all_of(value.assigned.begin(), value.assigned.end(),
                          base::identity())) {
            return std::optional<RowIndex>(value.assigned.size());
          } else {
            return std::optional<RowIndex>();
          }
        });
    for (const std::optional<RowIndex>& assigned_size : assigned_sizes) {
      CHECK(assigned_size.has_value()) << "length is missing";

      if (first_unused.has_value()) {
        CHECK_EQ(first_unused.value(), assigned_size.value())
            << "all table columns must have the same length";
      } else {
        first_unused = assigned_size;
      }
    }

    CHECK(first_unused.has_value())
        << "length is missing, maybe there are no table columns";

    // Record these columns so that we can prevent them from being used again.
    for (const auto& [column, value] : values) {
      lookup_table_columns_.push_back(column);
      // |value.default_value| must have value because we must have assigned
      // at least one cell in each column, and in that case we checked
      // that all cells up to |first_unused| were assigned.
      assignment_->FillFromRow(column.column(), first_unused.value(),
                               value.default_value.value().value());
    }
  }

 private:
  // not owned
  Assignment<F>* const assignment_;
  std::vector<LookupTableColumn>& lookup_table_columns_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_LOOKUP_TABLE_ASSIGNER_H_
