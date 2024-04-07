#ifndef TACHYON_ZK_PLONK_BASE_MULTI_PHASE_OWNED_TABLE_H_
#define TACHYON_ZK_PLONK_BASE_MULTI_PHASE_OWNED_TABLE_H_

#include <utility>
#include <vector>

#include "tachyon/zk/plonk/base/owned_table.h"

namespace tachyon::zk::plonk {

template <typename PolyOrEvals>
class MultiPhaseOwnedTable : public OwnedTable<PolyOrEvals> {
 public:
  using F = typename PolyOrEvals::Field;

  MultiPhaseOwnedTable() = default;
  MultiPhaseOwnedTable(std::vector<PolyOrEvals>&& fixed_columns,
                       std::vector<PolyOrEvals>&& advice_columns,
                       std::vector<PolyOrEvals>&& instance_columns,
                       absl::Span<const F> challenges)
      : OwnedTable<PolyOrEvals>(std::move(fixed_columns),
                                std::move(advice_columns),
                                std::move(instance_columns)),
        challenges_(challenges) {}

  void set_challenges(absl::Span<const F> challenges) {
    challenges_ = challenges;
  }
  absl::Span<const F> challenges() const { return challenges_; }

 private:
  absl::Span<const F> challenges_;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_BASE_MULTI_PHASE_OWNED_TABLE_H_
