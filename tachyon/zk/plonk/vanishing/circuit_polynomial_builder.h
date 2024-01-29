// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.
#ifndef TACHYON_ZK_PLONK_VANISHING_CIRCUIT_POLYNOMIAL_BUILDER_H_
#define TACHYON_ZK_PLONK_VANISHING_CIRCUIT_POLYNOMIAL_BUILDER_H_

#include <utility>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/numerics/checked_math.h"
#include "tachyon/base/parallelize.h"
#include "tachyon/zk/lookup/lookup_committed.h"
#include "tachyon/zk/plonk/circuit/column_key.h"
#include "tachyon/zk/plonk/circuit/owned_table.h"
#include "tachyon/zk/plonk/circuit/ref_table.h"
#include "tachyon/zk/plonk/circuit/rotation.h"
#include "tachyon/zk/plonk/permutation/permutation_committed.h"
#include "tachyon/zk/plonk/permutation/unpermuted_table.h"
#include "tachyon/zk/plonk/vanishing/evaluation_input.h"
#include "tachyon/zk/plonk/vanishing/graph_evaluator.h"
#include "tachyon/zk/plonk/vanishing/vanishing_utils.h"

namespace tachyon::zk {

template <typename PCS>
class ProvingKey;

// It generates "CircuitPolynomial" formed below:
// - gate₀(X) + y * gate₁(X) + ... + yⁱ * gateᵢ(X) + ...
// You can find more detailed theory in "Halo2 book"
// https://zcash.github.io/halo2/design/proving-system/vanishing.html
template <typename PCS>
class CircuitPolynomialBuilder {
 public:
  using F = typename PCS::Field;
  using Poly = typename PCS::Poly;
  using Evals = typename PCS::Evals;
  using Domain = typename PCS::Domain;
  using ExtendedDomain = typename PCS::ExtendedDomain;
  using ExtendedEvals = typename PCS::ExtendedEvals;

  CircuitPolynomialBuilder() = default;

  static CircuitPolynomialBuilder Create(
      const Domain* domain, const ExtendedDomain* extended_domain, size_t n,
      RowIndex blinding_factors, size_t cs_degree, const F* beta,
      const F* gamma, const F* theta, const F* y, const F* zeta,
      absl::Span<const F> challenges, const ProvingKey<PCS>* proving_key,
      const std::vector<PermutationCommitted<Poly>>* committed_permutations,
      const std::vector<std::vector<LookupCommitted<Poly>>>*
          committed_lookups_vec,
      const std::vector<RefTable<Poly>>* poly_tables) {
    CircuitPolynomialBuilder builder;
    builder.domain_ = domain;

    builder.n_ = static_cast<int32_t>(n);
    builder.num_parts_ = extended_domain->size() >> domain->log_size_of_group();
    builder.chunk_len_ = cs_degree - 2;

    builder.omega_ = &domain->group_gen();
    builder.extended_omega_ = &extended_domain->group_gen();
    builder.delta_ = GetDelta<F>();
    builder.beta_ = beta;
    builder.gamma_ = gamma;
    builder.theta_ = theta;
    builder.y_ = y;
    builder.zeta_ = zeta;
    builder.challenges_ = challenges;

    base::CheckedNumeric<int32_t> last_rotation = blinding_factors;
    builder.last_rotation_ = Rotation((-last_rotation - 1).ValueOrDie());
    builder.delta_start_ = *beta * *zeta;

    builder.proving_key_ = proving_key;
    builder.committed_permutations_ = committed_permutations;
    builder.committed_lookups_vec_ = committed_lookups_vec;
    builder.poly_tables_ = poly_tables;

    return builder;
  }

  void UpdateCurrentExtendedOmega() {
    current_extended_omega_ *= *extended_omega_;
  }

  // Returns an evaluation-formed polynomial as below.
  // - gate₀(X) + y * gate₁(X) + ... + yⁱ * gateᵢ(X) + ...
  ExtendedEvals BuildExtendedCircuitColumn(
      const GraphEvaluator<F>& custom_gate_evaluator,
      const std::vector<GraphEvaluator<F>>& lookup_evaluators) {
    std::vector<std::vector<F>> value_parts;
    value_parts.reserve(num_parts_);
    // Calculate the quotient polynomial for each part
    for (size_t i = 0; i < num_parts_; ++i) {
      UpdateVanishingProvingKey();

      std::vector<F> value_part =
          base::CreateVector(static_cast<size_t>(n_), F::Zero());
      size_t circuit_num = poly_tables_->size();
      for (size_t j = 0; j < circuit_num; ++j) {
        UpdateVanishingTable(j);
        UpdateValuesByCustomGates(custom_gate_evaluator, value_part);

        // Do iff there are permutation constraints.
        if ((*committed_permutations_)[j].product_polys().size() > 0) {
          UpdateVanishingPermutation(j);
          UpdateValuesByPermutation(value_part);
        }
        if ((*committed_lookups_vec_)[j].size() > 0) {
          UpdateVanishingLookups(j);
          UpdateValuesByLookups(lookup_evaluators, value_part);
        }
      }
      value_parts.push_back(std::move(value_part));
      UpdateCurrentExtendedOmega();
    }
    std::vector<F> extended =
        BuildExtendedColumnWithColumns(std::move(value_parts));
    return ExtendedEvals(std::move(extended));
  }

  void UpdateValuesByLookups(
      const std::vector<GraphEvaluator<F>>& lookup_evaluators,
      std::vector<F>& values) {
    for (size_t i = 0; i < committed_lookups_vec_->size(); ++i) {
      const GraphEvaluator<F>& ev = lookup_evaluators[i];

      base::Parallelize(values, [this, i, &ev](absl::Span<F> chunk,
                                               size_t chunk_offset,
                                               size_t chunk_size) {
        const Evals& input_coset = lookup_input_cosets_[i];
        const Evals& table_coset = lookup_table_cosets_[i];
        const Evals& product_coset = lookup_product_cosets_[i];

        EvaluationInput<Poly, Evals> evaluation_input = ExtractEvaluationInput(
            ev.CreateInitialIntermediates(), ev.CreateEmptyRotations());

        size_t start = chunk_offset * chunk_size;
        for (size_t j = 0; j < chunk.size(); ++j) {
          size_t idx = start + j;

          F zero = F::Zero();
          F table_value = ev.Evaluate(evaluation_input, idx, rot_scale_, zero);

          RowIndex r_next = Rotation(1).GetIndex(idx, rot_scale_, n_);
          RowIndex r_prev = Rotation(-1).GetIndex(idx, rot_scale_, n_);

          F a_minus_s = *input_coset[idx] - *table_coset[idx];

          // l_first(X) * (1 - z(X)) = 0
          chunk[j] *= *y_;
          chunk[j] += (one_ - *product_coset[idx]) * *l_first_[idx];

          // l_last(X) * (z(X)² - z(X)) = 0
          chunk[j] *= *y_;
          chunk[j] += (product_coset[idx]->Square() - *product_coset[idx]) *
                      *l_last_[idx];

          // clang-format off
          // A * (B - C) = 0 where
          //  - A = 1 - (l_last(X) + l_blind(X))
          //  - B = z(wX) * (a'(X) + β) * (s'(X) + γ)
          //  - C = z(X) * (θᵐ⁻¹ a₀(X) + ... + aₘ₋₁(X) + β) * (θᵐ⁻¹ s₀(X) + ... + sₘ₋₁(X) + γ)
          // clang-format on
          chunk[j] *= *y_;
          chunk[j] += (*product_coset[r_next] * (*input_coset[idx] + *beta_) *
                           (*table_coset[idx] + *gamma_) -
                       *product_coset[idx] * table_value) *
                      *l_active_row_[idx];

          // Check that the first values in the permuted input expression and
          // permuted fixed expression are the same.
          // l_first(X) * (a'(X) - s'(X)) = 0
          chunk[j] *= *y_;
          chunk[j] += a_minus_s * *l_first_[idx];

          // Check that each value in the permuted lookup input expression is
          // either equal to the value above it, or the value at the same
          // index in the permuted table expression. (1 - (l_last + l_blind)) *
          // (a′(X) − s′(X))⋅(a′(X) − a′(w⁻¹X)) = 0
          chunk[j] *= *y_;
          chunk[j] += a_minus_s * (*input_coset[idx] - *input_coset[r_prev]) *
                      *l_active_row_[idx];
        }
      });
    }
  }

  void UpdateValuesByPermutation(std::vector<F>& values) {
    base::Parallelize(values, [this](absl::Span<F> chunk, size_t chunk_offset,
                                     size_t chunk_size) {
      const std::vector<Evals>& product_cosets = permutation_product_cosets_;
      const std::vector<Evals>& cosets = permutation_cosets_;

      size_t start = chunk_offset * chunk_size;
      F beta_term = current_extended_omega_ * omega_->Pow(start);
      for (size_t i = 0; i < chunk.size(); ++i) {
        size_t idx = start + i;

        // Enforce only for the first set: l_first(X) * (1 - z₀(X)) = 0
        chunk[i] *= *y_;
        chunk[i] += (one_ - *product_cosets.front()[idx]) * *l_first_[idx];

        // Enforce only for the last set: l_last(X) * (z_l(X)² - z_l(X)) = 0
        const Evals& last_coset = product_cosets.back();
        chunk[i] *= *y_;
        chunk[i] +=
            *l_last_[idx] * (last_coset[idx]->Square() - *last_coset[idx]);

        // Except for the first set, enforce:
        // l_first(X) * (zᵢ(X) - zᵢ₋₁(w⁻¹X)) = 0
        RowIndex r_last = last_rotation_.GetIndex(idx, rot_scale_, n_);
        for (size_t set_idx = 0; set_idx < product_cosets.size(); ++set_idx) {
          if (set_idx == 0) continue;
          chunk[i] *= *y_;
          chunk[i] += *l_first_[idx] * (*product_cosets[set_idx][idx] -
                                        *product_cosets[set_idx - 1][r_last]);
        }

        // And for all the sets we enforce: (1 - (l_last(X) + l_blind(X))) *
        // (zᵢ(wX) * Πⱼ(p(X) + βsⱼ(X) + γ) - zᵢ(X) Πⱼ(p(X) + δʲβX + γ))
        F current_delta = delta_start_ * beta_term;
        RowIndex r_next = Rotation(1).GetIndex(idx, rot_scale_, n_);

        const std::vector<AnyColumnKey>& column_keys =
            proving_key_->verifying_key()
                .constraint_system()
                .permutation()
                .columns();
        std::vector<absl::Span<const AnyColumnKey>> column_key_chunks =
            base::ParallelizeMapByChunkSize(
                column_keys, chunk_len_,
                [](absl::Span<const AnyColumnKey> chunk) { return chunk; });
        std::vector<absl::Span<const Evals>> coset_chunks =
            base::ParallelizeMapByChunkSize(
                cosets, chunk_len_,
                [](absl::Span<const Evals> chunk) { return chunk; });

        for (size_t j = 0; j < product_cosets.size(); ++j) {
          std::vector<base::Ref<const Evals>> column_chunk =
              table_.GetColumns(column_key_chunks[j]);
          F left = CalculateLeft(column_chunk, coset_chunks[j], idx,
                                 product_cosets[j][r_next]);
          F right = CalculateRight(column_chunk, &current_delta, idx,
                                   product_cosets[j][idx]);
          chunk[i] *= *y_;
          chunk[i] += (left - right) * *l_active_row_[idx];
        }
        beta_term *= *omega_;
      }
    });
  }

 private:
  EvaluationInput<Poly, Evals> ExtractEvaluationInput(
      std ::vector<F>&& intermediates, std::vector<int32_t>&& rotations) {
    return EvaluationInput<Poly, Evals>(
        std::move(intermediates), std::move(rotations), &table_, challenges_,
        beta_, gamma_, theta_, y_, n_);
  }

  template <typename Evals>
  F CalculateLeft(const std::vector<base::Ref<const Evals>>& column_chunk,
                  absl::Span<const Evals> coset_chunk, size_t idx,
                  const F* initial_value) {
    F left = *initial_value;
    for (size_t i = 0; i < column_chunk.size(); ++i) {
      left *=
          *(*column_chunk[i])[idx] + *beta_ * *coset_chunk[i][idx] + *gamma_;
    }
    return left;
  }

  template <typename Evals>
  F CalculateRight(const std::vector<base::Ref<const Evals>>& column_chunk,
                   F* current_delta, size_t idx, const F* initial_value) {
    F right = *initial_value;
    for (size_t i = 0; i < column_chunk.size(); ++i) {
      right *= *(*column_chunk[i])[idx] + *current_delta + *gamma_;
      *current_delta *= delta_;
    }
    return right;
  }

  void UpdateValuesByCustomGates(const GraphEvaluator<F>& custom_gate_evaluator,
                                 std::vector<F>& values) {
    base::Parallelize(values, [this, &custom_gate_evaluator](
                                  absl::Span<F> chunk, size_t chunk_offset,
                                  size_t chunk_size) {
      EvaluationInput<Poly, Evals> evaluation_input = ExtractEvaluationInput(
          custom_gate_evaluator.CreateInitialIntermediates(),
          custom_gate_evaluator.CreateEmptyRotations());

      size_t start = chunk_offset * chunk_size;
      for (size_t i = 0; i < chunk.size(); ++i) {
        chunk[i] = custom_gate_evaluator.Evaluate(evaluation_input, start + i,
                                                  rot_scale_, chunk[i]);
      }
    });
  }

  void UpdateVanishingProvingKey() {
    l_first_ = CoeffToExtendedPart(domain_, proving_key_->l_first(), *zeta_,
                                   current_extended_omega_);
    l_last_ = CoeffToExtendedPart(domain_, proving_key_->l_last(), *zeta_,
                                  current_extended_omega_);
    l_active_row_ = CoeffToExtendedPart(domain_, proving_key_->l_active_row(),
                                        *zeta_, current_extended_omega_);
  }

  void UpdateVanishingPermutation(size_t circuit_idx) {
    permutation_product_cosets_ = CoeffsToExtendedPart(
        domain_,
        absl::MakeConstSpan(
            (*committed_permutations_)[circuit_idx].product_polys()),
        *zeta_, current_extended_omega_);
    permutation_cosets_ = CoeffsToExtendedPart(
        domain_,
        absl::MakeConstSpan(proving_key_->permutation_proving_key().polys()),
        *zeta_, current_extended_omega_);
  }

  void UpdateVanishingLookups(size_t circuit_idx) {
    size_t num_lookups = committed_lookups_vec_->size();
    std::vector<LookupCommitted<Poly>> current_committed_lookups =
        (*committed_lookups_vec_)[circuit_idx];
    lookup_product_cosets_.clear();
    lookup_input_cosets_.clear();
    lookup_table_cosets_.clear();
    lookup_product_cosets_.reserve(num_lookups);
    lookup_input_cosets_.reserve(num_lookups);
    lookup_table_cosets_.reserve(num_lookups);
    for (size_t i = 0; i < num_lookups; ++i) {
      lookup_product_cosets_.push_back(CoeffToExtendedPart(
          domain_, current_committed_lookups[i].product_poly(), *zeta_,
          current_extended_omega_));
      lookup_input_cosets_.push_back(CoeffToExtendedPart(
          domain_, current_committed_lookups[i].permuted_input_poly(), *zeta_,
          current_extended_omega_));
      lookup_table_cosets_.push_back(CoeffToExtendedPart(
          domain_, current_committed_lookups[i].permuted_table_poly(), *zeta_,
          current_extended_omega_));
    }
  }

  void UpdateVanishingTable(size_t circuit_idx) {
    std::vector<Evals> fixed_columns = CoeffsToExtendedPart(
        domain_, (*poly_tables_)[circuit_idx].GetFixedColumns(), *zeta_,
        current_extended_omega_);
    std::vector<Evals> advice_columns = CoeffsToExtendedPart(
        domain_, (*poly_tables_)[circuit_idx].GetAdviceColumns(), *zeta_,
        current_extended_omega_);
    std::vector<Evals> instance_columns = CoeffsToExtendedPart(
        domain_, (*poly_tables_)[circuit_idx].GetInstanceColumns(), *zeta_,
        current_extended_omega_);
    table_ =
        OwnedTable<Evals>(std::move(fixed_columns), std::move(advice_columns),
                          std::move(instance_columns));
  }

  // not owned
  const Domain* domain_ = nullptr;

  F one_ = F::One();
  F current_extended_omega_ = F::One();
  size_t rot_scale_ = 1;

  int32_t n_ = 0;
  size_t num_parts_ = 0;
  size_t chunk_len_ = 0;
  // not owned
  const F* omega_ = nullptr;
  // not owned
  const F* extended_omega_ = nullptr;
  F delta_;
  // not owned
  const F* beta_ = nullptr;
  // not owned
  const F* gamma_ = nullptr;
  // not owned
  const F* theta_ = nullptr;
  // not owned
  const F* y_ = nullptr;
  // not owned
  const F* zeta_ = nullptr;
  absl::Span<const F> challenges_;
  Rotation last_rotation_;
  F delta_start_;

  // not owned
  const ProvingKey<PCS>* proving_key_;
  // not owned
  const std::vector<PermutationCommitted<Poly>>* committed_permutations_;
  // not owned
  const std::vector<std::vector<LookupCommitted<Poly>>>* committed_lookups_vec_;
  // not owned
  const std::vector<RefTable<Poly>>* poly_tables_;

  Evals l_first_;
  Evals l_last_;
  Evals l_active_row_;

  std::vector<Evals> permutation_product_cosets_;
  std::vector<Evals> permutation_cosets_;

  std::vector<Evals> lookup_product_cosets_;
  std::vector<Evals> lookup_input_cosets_;
  std::vector<Evals> lookup_table_cosets_;

  OwnedTable<Evals> table_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_VANISHING_CIRCUIT_POLYNOMIAL_BUILDER_H_
