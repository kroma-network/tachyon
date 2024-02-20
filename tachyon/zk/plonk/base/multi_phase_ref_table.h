#ifndef TACHYON_ZK_PLONK_BASE_MULTI_PHASE_REF_TABLE_H_
#define TACHYON_ZK_PLONK_BASE_MULTI_PHASE_REF_TABLE_H_

#include "tachyon/zk/plonk/base/ref_table.h"

namespace tachyon::zk::plonk {

template <typename PolyOrEvals>
class MultiPhaseRefTable : public RefTable<PolyOrEvals> {
 public:
  using F = typename PolyOrEvals::Field;

  MultiPhaseRefTable() = default;
  MultiPhaseRefTable(absl::Span<const PolyOrEvals> fixed_columns,
                     absl::Span<const PolyOrEvals> advice_columns,
                     absl::Span<const PolyOrEvals> instance_columns,
                     absl::Span<const F> challenges)
      : RefTable<PolyOrEvals>(fixed_columns, advice_columns, instance_columns),
        challenges_(challenges) {}

  absl::Span<const F> challenges() const { return challenges_; }

 private:
  absl::Span<const F> challenges_;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_BASE_MULTI_PHASE_REF_TABLE_H_
