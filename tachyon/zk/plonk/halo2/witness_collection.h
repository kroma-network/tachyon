// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_HALO2_WITNESS_COLLECTION_H_
#define TACHYON_ZK_PLONK_HALO2_WITNESS_COLLECTION_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/range.h"
#include "tachyon/zk/plonk/base/phase.h"
#include "tachyon/zk/plonk/layout/assignment.h"

namespace tachyon::zk::plonk::halo2 {

template <typename Evals, typename RationalEvals>
class WitnessCollection : public Assignment<typename Evals::Field> {
 public:
  using F = typename Evals::Field;
  using AssignCallback = typename Assignment<F>::AssignCallback;

  WitnessCollection() = default;
  template <typename Domain>
  WitnessCollection(const Domain* domain, size_t num_advice_columns,
                    RowIndex usable_rows, Phase current_phase,
                    const absl::btree_map<size_t, F>& challenges,
                    const std::vector<Evals>& instance_columns)
      : advices_(base::CreateVector(num_advice_columns,
                                    domain->template Empty<RationalEvals>())),
        usable_rows_(base::Range<RowIndex>::Until(usable_rows)),
        current_phase_(current_phase),
        challenges_(challenges),
        instance_columns_(instance_columns) {}

  // NOTE(dongchangYoo): This getter of |advices| transfers ownership as well.
  // That's why, |WitnessCollection| will be released as soon as emitting it.
  std::vector<RationalEvals>&& TakeAdvices() && { return std::move(advices_); }

  Value<F> QueryInstance(const InstanceColumnKey& column,
                         RowIndex row) override {
    CHECK(usable_rows_.Contains(row));
    CHECK_LT(column.index(), instance_columns_.size());

    return Value<F>::Known(*instance_columns_[column.index()][row]);
  }

  void AssignAdvice(std::string_view, const AdviceColumnKey& column,
                    RowIndex row, AssignCallback assign) override {
    // Ignore assignment of advice column in different phase than current one.
    if (current_phase_ < column.phase()) return;

    CHECK(usable_rows_.Contains(row));
    CHECK_LT(column.index(), advices_.size());

    *advices_[column.index()][row] = std::move(assign).Run().value();
  }

  Value<F> GetChallenge(Challenge challenge) override {
    CHECK_LT(challenge.index(), challenges_.size());
    return Value<F>::Known(challenges_[challenge.index()]);
  }

 private:
  std::vector<RationalEvals> advices_;
  base::Range<RowIndex> usable_rows_;
  Phase current_phase_;
  absl::btree_map<size_t, F> challenges_;
  std::vector<Evals> instance_columns_;
};

}  // namespace tachyon::zk::plonk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_WITNESS_COLLECTION_H_
