// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_VANISHING_CIRCUIT_POLYNOMIAL_BUILDER_H_
#define TACHYON_ZK_PLONK_VANISHING_CIRCUIT_POLYNOMIAL_BUILDER_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/base/containers/adapters.h"
#include "tachyon/base/numerics/checked_math.h"
#include "tachyon/base/parallelize.h"
#include "tachyon/base/types/always_false.h"
#include "tachyon/zk/base/rotation.h"
#include "tachyon/zk/lookup/halo2/evaluator.h"
#include "tachyon/zk/lookup/halo2/prover.h"
#include "tachyon/zk/lookup/log_derivative_halo2/evaluator.h"
#include "tachyon/zk/plonk/base/column_key.h"
#include "tachyon/zk/plonk/base/owned_table.h"
#include "tachyon/zk/plonk/base/ref_table.h"
#include "tachyon/zk/plonk/keys/proving_key_forward.h"
#include "tachyon/zk/plonk/permutation/permutation_prover.h"
#include "tachyon/zk/plonk/vanishing/evaluation_input.h"
#include "tachyon/zk/plonk/vanishing/graph_evaluator.h"
#include "tachyon/zk/plonk/vanishing/vanishing_utils.h"

namespace tachyon::zk::plonk {

// It generates "CircuitPolynomial" formed below:
// - gate₀(X) + y * gate₁(X) + ... + yⁱ * gateᵢ(X) + ...
// You can find more detailed theory in "Halo2 book"
// https://zcash.github.io/halo2/design/proving-system/vanishing.html
template <typename PCS, typename LS>
class CircuitPolynomialBuilder {
 public:
  using F = typename PCS::Field;
  using C = typename PCS::Commitment;
  using Poly = typename PCS::Poly;
  using Evals = typename PCS::Evals;
  using Domain = typename PCS::Domain;
  using ExtendedDomain = typename PCS::ExtendedDomain;
  using ExtendedEvals = typename PCS::ExtendedEvals;
  using LookupProver = typename LS::Prover;
  using LookupEvaluator = typename LS::Evaluator;

  CircuitPolynomialBuilder(
      const F& omega, const F& extended_omega, const F& theta, const F& beta,
      const F& gamma, const F& y, const F& zeta,
      const ProvingKey<LS>& proving_key,
      const std::vector<PermutationProver<Poly, Evals>>& permutation_provers,
      const std::vector<LookupProver>& lookup_provers,
      const std::vector<MultiPhaseRefTable<Poly>>& poly_tables)
      : omega_(omega),
        extended_omega_(extended_omega),
        theta_(theta),
        beta_(beta),
        gamma_(gamma),
        y_(y),
        zeta_(zeta),
        proving_key_(proving_key),
        permutation_provers_(permutation_provers),
        lookup_provers_(lookup_provers),
        poly_tables_(poly_tables) {}

  static CircuitPolynomialBuilder Create(
      const Domain* domain, const ExtendedDomain* extended_domain, size_t n,
      RowOffset last_row, size_t cs_degree,
      const std::vector<MultiPhaseRefTable<Poly>>& poly_tables, const F& theta,
      const F& beta, const F& gamma, const F& y, const F& zeta,
      const ProvingKey<LS>& proving_key,
      const std::vector<PermutationProver<Poly, Evals>>& permutation_provers,
      const std::vector<LookupProver>& lookup_provers) {
    CircuitPolynomialBuilder builder(
        domain->group_gen(), extended_domain->group_gen(), theta, beta, gamma,
        y, zeta, proving_key, permutation_provers, lookup_provers, poly_tables);
    builder.domain_ = domain;

    builder.n_ = static_cast<int32_t>(n);
    builder.num_parts_ = extended_domain->size() >> domain->log_size_of_group();
    builder.chunk_len_ = cs_degree - 2;

    builder.delta_ = GetDelta<F>();

    builder.last_rotation_ = Rotation(last_row);
    builder.delta_start_ = beta * zeta;
    return builder;
  }

  void UpdateCurrentExtendedOmega() {
    current_extended_omega_ *= extended_omega_;
  }

  // Returns an evaluation-formed polynomial as below.
  // - gate₀(X) + y * gate₁(X) + ... + yⁱ * gateᵢ(X) + ...
  ExtendedEvals BuildExtendedCircuitColumn(
      const GraphEvaluator<F>& custom_gate_evaluator,
      LookupEvaluator& lookup_evaluator) {
    std::vector<std::vector<F>> value_parts;
    value_parts.reserve(num_parts_);
    // Calculate the quotient polynomial for each part
    for (size_t i = 0; i < num_parts_; ++i) {
      VLOG(1) << "BuildExtendedCircuitColumn part: (" << i + 1 << " / "
              << num_parts_ << ")";

      coset_domain_ = domain_->GetCoset(zeta_ * current_extended_omega_);

      UpdateLPolys();

      std::vector<F> value_part(static_cast<size_t>(n_));
      size_t circuit_num = poly_tables_.size();
      for (size_t j = 0; j < circuit_num; ++j) {
        VLOG(1) << "BuildExtendedCircuitColumn part: " << i << " circuit: ("
                << j + 1 << " / " << circuit_num << ")";
        UpdateTable(j);
        // Do iff there are permutation constraints.
        if (permutation_provers_[j].grand_product_polys().size() > 0)
          UpdatePermutationCosets(j);
        // Do iff there are lookup constraints.
        size_t lookup_polys_num = 0;
        if constexpr (LS::type == lookup::Type::kHalo2) {
          lookup_polys_num = lookup_provers_[j].grand_product_polys().size();
        } else if constexpr (LS::type == lookup::Type::kLogDerivativeHalo2) {
          lookup_polys_num = lookup_provers_[j].grand_sum_polys().size();
        } else {
          static_assert(base::AlwaysFalse<LS>);
        }

        if (lookup_polys_num > 0) lookup_evaluator.UpdateLookupCosets(*this, j);
        base::Parallelize(
            value_part,
            [this, &custom_gate_evaluator, &lookup_evaluator](
                absl::Span<F> chunk, size_t chunk_offset, size_t chunk_size) {
              UpdateChunkByCustomGates(custom_gate_evaluator, chunk,
                                       chunk_offset, chunk_size);
              UpdateChunkByPermutation(chunk, chunk_offset, chunk_size);
              lookup_evaluator.UpdateChunkByLookups(*this, chunk, chunk_offset,
                                                    chunk_size);
            });
      }

      value_parts.push_back(std::move(value_part));
      UpdateCurrentExtendedOmega();
    }
    std::vector<F> extended = BuildExtendedColumnWithColumns(value_parts);
    return ExtendedEvals(std::move(extended));
  }

  void UpdateChunkByPermutation(absl::Span<F> chunk, size_t chunk_offset,
                                size_t chunk_size) {
    if (permutation_product_cosets_.empty()) return;

    const std::vector<Evals>& product_cosets = permutation_product_cosets_;
    const std::vector<Evals>& cosets = permutation_cosets_;

    const std::vector<AnyColumnKey>& column_keys = proving_key_.verifying_key()
                                                       .constraint_system()
                                                       .permutation()
                                                       .columns();
    std::vector<std::vector<base::Ref<const Evals>>> column_chunks =
        base::Map(base::Chunked(column_keys, chunk_len_),
                  [this](absl::Span<const AnyColumnKey> column_key_chunk) {
                    return table_.GetColumns(column_key_chunk);
                  });
    std::vector<absl::Span<const Evals>> coset_chunks =
        base::Map(base::Chunked(cosets, chunk_len_),
                  [](absl::Span<const Evals> chunk) { return chunk; });

    size_t start = chunk_offset * chunk_size;
    F beta_term = current_extended_omega_ * omega_.Pow(start);
    for (size_t i = 0; i < chunk.size(); ++i) {
      size_t idx = start + i;

      // Enforce only for the first set: l_first(X) * (1 - z₀(X)) = 0
      chunk[i] *= y_;
      chunk[i] += (one_ - product_cosets.front()[idx]) * l_first_[idx];

      // Enforce only for the last set: l_last(X) * (z_l(X)² - z_l(X)) = 0
      const Evals& last_coset = product_cosets.back();
      chunk[i] *= y_;
      chunk[i] += l_last_[idx] * (last_coset[idx].Square() - last_coset[idx]);

      // Except for the first set, enforce:
      // l_first(X) * (zⱼ(X) - zⱼ₋₁(w⁻¹X)) = 0
      RowIndex r_last = last_rotation_.GetIndex(idx, /*scale=*/1, n_);
      for (size_t j = 0; j < product_cosets.size(); ++j) {
        if (j == 0) continue;
        chunk[i] *= y_;
        chunk[i] += l_first_[idx] *
                    (product_cosets[j][idx] - product_cosets[j - 1][r_last]);
      }

      // And for all the sets we enforce: (1 - (l_last(X) + l_blind(X))) *
      // (zⱼ(wX) * Πⱼ(p(X) + βsⱼ(X) + γ) - zⱼ(X) Πⱼ(p(X) + δʲβX + γ))
      F current_delta = delta_start_ * beta_term;
      RowIndex r_next = Rotation(1).GetIndex(idx, /*scale=*/1, n_);

      for (size_t j = 0; j < product_cosets.size(); ++j) {
        F left = CalculateLeft(column_chunks[j], coset_chunks[j], idx,
                               product_cosets[j][r_next]);
        F right = CalculateRight(column_chunks[j], current_delta, idx,
                                 product_cosets[j][idx]);
        chunk[i] *= y_;
        chunk[i] += (left - right) * l_active_row_[idx];
      }
      beta_term *= omega_;
    }
  }

 private:
  friend class lookup::halo2::Evaluator<Evals>;
  friend class lookup::log_derivative_halo2::Evaluator<Evals>;

  EvaluationInput<Evals> ExtractEvaluationInput(
      std ::vector<F>&& intermediates, std::vector<int32_t>&& rotations) {
    return EvaluationInput<Evals>(std::move(intermediates),
                                  std::move(rotations), table_, theta_, beta_,
                                  gamma_, y_, n_);
  }

  template <typename Evals>
  F CalculateLeft(const std::vector<base::Ref<const Evals>>& column_chunk,
                  absl::Span<const Evals> coset_chunk, size_t idx,
                  const F& initial_value) {
    F left = initial_value;
    for (size_t i = 0; i < column_chunk.size(); ++i) {
      left *= (*column_chunk[i])[idx] + beta_ * coset_chunk[i][idx] + gamma_;
    }
    return left;
  }

  template <typename Evals>
  F CalculateRight(const std::vector<base::Ref<const Evals>>& column_chunk,
                   F& current_delta, size_t idx, const F& initial_value) {
    F right = initial_value;
    for (size_t i = 0; i < column_chunk.size(); ++i) {
      right *= (*column_chunk[i])[idx] + current_delta + gamma_;
      current_delta *= delta_;
    }
    return right;
  }

  void UpdateChunkByCustomGates(const GraphEvaluator<F>& custom_gate_evaluator,
                                absl::Span<F> chunk, size_t chunk_offset,
                                size_t chunk_size) {
    EvaluationInput<Evals> evaluation_input = ExtractEvaluationInput(
        custom_gate_evaluator.CreateInitialIntermediates(),
        custom_gate_evaluator.CreateEmptyRotations());
    size_t start = chunk_offset * chunk_size;
    for (size_t i = 0; i < chunk.size(); ++i) {
      chunk[i] = custom_gate_evaluator.Evaluate(evaluation_input, start + i,
                                                /*scale=*/1, chunk[i]);
    }
  }

  void UpdateLPolys() {
    l_first_ = coset_domain_->FFT(proving_key_.l_first());
    l_last_ = coset_domain_->FFT(proving_key_.l_last());
    l_active_row_ = coset_domain_->FFT(proving_key_.l_active_row());
  }

  void UpdatePermutationCosets(size_t circuit_idx) {
    const std::vector<BlindedPolynomial<Poly, Evals>>& grand_product_polys =
        permutation_provers_[circuit_idx].grand_product_polys();
    permutation_product_cosets_.resize(grand_product_polys.size());
    for (size_t i = 0; i < grand_product_polys.size(); ++i) {
      permutation_product_cosets_[i] =
          coset_domain_->FFT(grand_product_polys[i].poly());
    }

    const std::vector<Poly>& polys =
        proving_key_.permutation_proving_key().polys();
    permutation_cosets_.resize(polys.size());
    for (size_t i = 0; i < polys.size(); ++i) {
      permutation_cosets_[i] = coset_domain_->FFT(polys[i]);
    }
  }

  void UpdateTable(size_t circuit_idx) {
    absl::Span<const Poly> new_fixed_columns =
        poly_tables_[circuit_idx].GetFixedColumns();
    std::vector<Evals>& fixed_columns = table_.fixed_columns();
    fixed_columns.resize(new_fixed_columns.size());
    for (size_t i = 0; i < new_fixed_columns.size(); ++i) {
      fixed_columns[i] = coset_domain_->FFT(new_fixed_columns[i]);
    }

    absl::Span<const Poly> new_advice_columns =
        poly_tables_[circuit_idx].GetAdviceColumns();
    std::vector<Evals>& advice_columns = table_.advice_columns();
    advice_columns.resize(new_advice_columns.size());
    for (size_t i = 0; i < new_advice_columns.size(); ++i) {
      advice_columns[i] = coset_domain_->FFT(new_advice_columns[i]);
    }

    absl::Span<const Poly> new_instance_columns =
        poly_tables_[circuit_idx].GetInstanceColumns();
    std::vector<Evals>& instance_columns = table_.instance_columns();
    instance_columns.resize(new_instance_columns.size());
    for (size_t i = 0; i < new_instance_columns.size(); ++i) {
      instance_columns[i] = coset_domain_->FFT(new_instance_columns[i]);
    }

    table_.set_challenges(poly_tables_[circuit_idx].challenges());
  }

  // not owned
  const Domain* domain_ = nullptr;
  std::unique_ptr<Domain> coset_domain_;

  F one_ = F::One();
  F current_extended_omega_ = F::One();

  int32_t n_ = 0;
  size_t num_parts_ = 0;
  size_t chunk_len_ = 0;
  const F& omega_;
  const F& extended_omega_;
  F delta_;
  const F& theta_;
  const F& beta_;
  const F& gamma_;
  const F& y_;
  const F& zeta_;
  Rotation last_rotation_;
  F delta_start_;

  const ProvingKey<LS>& proving_key_;
  const std::vector<PermutationProver<Poly, Evals>>& permutation_provers_;
  const std::vector<LookupProver>& lookup_provers_;
  const std::vector<MultiPhaseRefTable<Poly>>& poly_tables_;

  Evals l_first_;
  Evals l_last_;
  Evals l_active_row_;

  std::vector<Evals> permutation_product_cosets_;
  std::vector<Evals> permutation_cosets_;

  std::vector<Evals> lookup_product_cosets_;
  std::vector<Evals> lookup_input_cosets_;
  std::vector<Evals> lookup_table_cosets_;

  MultiPhaseOwnedTable<Evals> table_;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_VANISHING_CIRCUIT_POLYNOMIAL_BUILDER_H_
