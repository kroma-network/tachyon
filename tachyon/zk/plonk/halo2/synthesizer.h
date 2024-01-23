// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_HALO2_SYNTHESIZER_H_
#define TACHYON_ZK_PLONK_HALO2_SYNTHESIZER_H_

#include <utility>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/zk/base/entities/prover_base.h"
#include "tachyon/zk/plonk/constraint_system.h"
#include "tachyon/zk/plonk/halo2/witness_collection.h"

namespace tachyon::zk::halo2 {

template <typename PCS>
class Synthesizer {
 public:
  using F = typename PCS::Field;
  using Poly = typename PCS::Poly;
  using Evals = typename PCS::Evals;
  using RationalEvals = typename PCS::RationalEvals;

  Synthesizer() = default;
  Synthesizer(size_t num_circuits, const ConstraintSystem<F>* constraint_system)
      : num_circuits_(num_circuits), constraint_system_(constraint_system) {
    advice_columns_vec_.resize(num_circuits_);
    advice_blinds_vec_.resize(num_circuits_);
    for (size_t i = 0; i < num_circuits_; ++i) {
      // And these may be assigned with random order.
      advice_columns_vec_[i] = base::CreateVector(
          constraint_system->num_advice_columns(), Evals::Zero());
      advice_blinds_vec_[i] = base::CreateVector(
          constraint_system->num_advice_columns(), F::Zero());
    }
  }

  // Synthesize circuit and store advice columns.
  template <typename Circuit>
  void GenerateAdviceColumns(
      ProverBase<PCS>* prover, std::vector<Circuit>& circuits,
      const std::vector<std::vector<Evals>>& instance_columns_vec) {
    CHECK_EQ(num_circuits_, circuits.size());

    ConstraintSystem<F> empty_constraint_system;
    typename Circuit::Config config =
        Circuit::Configure(empty_constraint_system);

    for (Phase current_phase : constraint_system_->GetPhases()) {
      for (size_t i = 0; i < num_circuits_; ++i) {
        std::vector<RationalEvals> rational_advice_columns =
            GenerateRationalAdvices(prover, current_phase,
                                    instance_columns_vec[i], circuits[i],
                                    config);

        // Parse only indices related to the |current_phase|.
        const std::vector<Phase>& advice_phases =
            constraint_system_->advice_column_phases();
        if constexpr (PCS::kSupportsBatchMode) {
          prover->pcs().SetBatchMode(rational_advice_columns.size());
        }
        for (size_t j = 0; j < rational_advice_columns.size(); ++j) {
          if (current_phase != advice_phases[j]) continue;
          const RationalEvals& column = rational_advice_columns[j];
          std::vector<F> evaluated;
          CHECK(math::RationalField<F>::BatchEvaluate(column.evaluations(),
                                                      &evaluated));
          // Add blinding factors to advice columns
          evaluated[prover->pcs().N() - 1] = F::One();

          Evals evaluated_evals(std::move(evaluated));
          if constexpr (PCS::kSupportsBatchMode) {
            prover->BatchCommitAt(evaluated_evals, j);
          } else {
            prover->CommitAndWriteToProof(evaluated_evals);
          }
          SetAdviceColumn(i, j, std::move(evaluated_evals),
                          prover->blinder().Generate());
        }
        if constexpr (PCS::kSupportsBatchMode) {
          prover->RetrieveAndWriteBatchCommitmentsToProof();
        }
      }
      UpdateChallenges(prover, current_phase);
    }
  }

  // Return |challenge_| as a vector.
  std::vector<F> ExportChallenges() {
    return base::Map(
        std::make_move_iterator(challenges_.begin()),
        std::make_move_iterator(challenges_.end()),
        [](std::pair<size_t, F>&& item) { return std::move(item.second); });
  }

  // Move out |challenge_| as a vector.
  std::vector<F> TakeChallenges() && {
    return base::Map(
        std::make_move_iterator(challenges_.begin()),
        std::make_move_iterator(challenges_.end()),
        [](std::pair<size_t, F>&& item) { return std::move(item.second); });
  }
  std::vector<std::vector<Evals>>&& TakeAdviceColumnsVec() && {
    return std::move(advice_columns_vec_);
  };
  std::vector<std::vector<F>>&& TakeAdviceBlindsVec() && {
    return std::move(advice_blinds_vec_);
  };

 private:
  void SetAdviceColumn(size_t circuit_idx, size_t column_idx, Evals&& column,
                       F&& blind) {
    CHECK_LT(circuit_idx, num_circuits_);
    CHECK_LT(column_idx, constraint_system_->num_advice_columns());
    advice_columns_vec_[circuit_idx][column_idx] = std::move(column);
    advice_blinds_vec_[circuit_idx][column_idx] = std::move(blind);
  }

  // Performs synthesis for a specific |circuit| and a specific |phase|, and
  // returns a vector of |RationalEvals|.
  template <typename Circuit>
  std::vector<RationalEvals> GenerateRationalAdvices(
      ProverBase<PCS>* prover, const Phase phase,
      const std::vector<Evals>& instance_columns, Circuit& circuit,
      const typename Circuit::Config& config) {
    // The prover will not be allowed to assign values to advice
    // cells that exist within inactive rows, which include some
    // number of blinding factors and an extra row for use in the
    // permutation argument.
    WitnessCollection<PCS> witness(
        prover->domain(), constraint_system_->num_advice_columns(),
        prover->GetUsableRows(), phase, challenges_, instance_columns);

    typename Circuit::FloorPlanner floor_planner;
    floor_planner.Synthesize(&witness, circuit, config.Clone(),
                             constraint_system_->constants());

    return std::move(witness).TakeAdvices();
  }

  void UpdateChallenges(ProverBase<PCS>* prover, const Phase phase) {
    const std::vector<Phase>& phases = constraint_system_->challenge_phases();
    for (size_t i = 0; i < phases.size(); ++i) {
      if (phase == phases[i]) {
        auto it =
            challenges_.try_emplace(i, prover->GetWriter()->SqueezeChallenge());
        CHECK(it.second);
      }
    }
  }

  size_t num_circuits_ = 0;
  // not owned
  const ConstraintSystem<F>* constraint_system_ = nullptr;

  absl::btree_map<size_t, F> challenges_;
  std::vector<std::vector<Evals>> advice_columns_vec_;
  std::vector<std::vector<F>> advice_blinds_vec_;
};

}  // namespace tachyon::zk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_SYNTHESIZER_H_
