// Copyright (c) 2022-2024 Scroll
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.scroll and the LICENCE-APACHE.scroll
// file.

#ifndef TACHYON_ZK_LOOKUP_LOG_DERIVATIVE_HALO2_EVALUATOR_H_
#define TACHYON_ZK_LOOKUP_LOG_DERIVATIVE_HALO2_EVALUATOR_H_

#include <memory>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include "tachyon/zk/lookup/log_derivative_halo2/prover.h"
#include "tachyon/zk/plonk/vanishing/circuit_polynomial_builder_forward.h"
#include "tachyon/zk/plonk/vanishing/graph_evaluator.h"

namespace tachyon::zk::lookup::log_derivative_halo2 {

template <typename F>
struct LookupEvaluatorsPair {
  std::vector<plonk::GraphEvaluator<F>> inputs_evaluator;
  plonk::GraphEvaluator<F> table_evaluator;

  LookupEvaluatorsPair(std::vector<plonk::GraphEvaluator<F>>&& inputs_evaluator,
                       plonk::GraphEvaluator<F>&& table_evaluator)
      : inputs_evaluator(std::move(inputs_evaluator)),
        table_evaluator(std::move(table_evaluator)) {}
};

template <typename Evals>
class Evaluator {
 public:
  using F = typename Evals::Field;

  const std::vector<LookupEvaluatorsPair<F>>& lookup_evaluators_pairs() const {
    return lookup_evaluators_pairs_;
  }

  void EvaluateLookups(const std::vector<Argument<F>>& lookups) {
    lookup_evaluators_pairs_.reserve(lookups.size());
    for (const Argument<F>& lookup : lookups) {
      plonk::GraphEvaluator<F> graph_table;
      std::vector<plonk::GraphEvaluator<F>> graph_inputs(
          lookup.inputs_expressions().size());

      auto compress =
          [](plonk::GraphEvaluator<F>& graph,
             const std::vector<std::unique_ptr<Expression<F>>>& expressions) {
            std::vector<plonk::ValueSource> parts = base::Map(
                expressions,
                [&graph](const std::unique_ptr<Expression<F>>& expression) {
                  return graph.AddExpression(expression.get());
                });
            return graph.AddCalculation(plonk::Calculation::Horner(
                plonk::ValueSource::ZeroConstant(), std::move(parts),
                plonk::ValueSource::Theta()));
          };

      for (size_t i = 0; i < lookup.inputs_expressions().size(); ++i) {
        // f_compressed(X) = θᵐ⁻¹f₀(X) + θᵐ⁻²f₁(X) + ... + θfₘ₋₂(X) + fₘ₋₁(X)
        plonk::ValueSource compressed_input_coset =
            compress(graph_inputs[i], lookup.inputs_expressions()[i]);
        // f_compressed(X) + β
        graph_inputs[i].AddCalculation(plonk::Calculation::Add(
            compressed_input_coset, plonk::ValueSource::Beta()));
      }
      // t_compressed(X) = θᵐ⁻¹t₀(X) + θᵐ⁻²t₁(X) + ... + θtₘ₋₂(X) + tₘ₋₁(X)
      plonk::ValueSource compressed_table_coset =
          compress(graph_table, lookup.table_expressions());
      // t_compressed(X) + β
      graph_table.AddCalculation(plonk::Calculation::Add(
          compressed_table_coset, plonk::ValueSource::Beta()));

      lookup_evaluators_pairs_.push_back(LookupEvaluatorsPair<F>(
          std::move(graph_inputs), std::move(graph_table)));
    }
  }

  template <plonk::halo2::Vendor Vendor, typename PCS, typename LS>
  void UpdateChunkByLookups(
      plonk::CircuitPolynomialBuilder<Vendor, PCS, LS>& builder,
      absl::Span<F> chunk, size_t chunk_offset, size_t chunk_size) {
    for (size_t lookup_idx = 0; lookup_idx < lookup_evaluators_pairs_.size();
         ++lookup_idx) {
      const std::vector<plonk::GraphEvaluator<F>>& inputs_evaluator =
          lookup_evaluators_pairs_[lookup_idx].inputs_evaluator;
      const plonk::GraphEvaluator<F>& table_evaluator =
          lookup_evaluators_pairs_[lookup_idx].table_evaluator;
      const Evals& sum_coset = lookup_sum_cosets_[lookup_idx];
      const Evals& m_coset = lookup_m_cosets_[lookup_idx];

      std::vector<plonk::EvaluationInput<Evals>> inputs_eval_data = base::Map(
          inputs_evaluator,
          [&builder](const plonk::GraphEvaluator<F>& input_evaluator) {
            return builder.ExtractEvaluationInput(
                input_evaluator.CreateInitialIntermediates(),
                input_evaluator.CreateEmptyRotations());
          });

      plonk::EvaluationInput<Evals> table_eval_data =
          builder.ExtractEvaluationInput(
              table_evaluator.CreateInitialIntermediates(),
              table_evaluator.CreateEmptyRotations());

      // φᵢ(X) = fᵢ(X) + β
      // τ(X) = t(X) + β
      // LHS = τ(X) * Π(φᵢ(X)) * (ϕ(ω * X) - ϕ(X))
      // RHS = τ(X) * Π(φᵢ(X)) * (Σ 1/(φᵢ(X)) - m(X) / τ(X))))
      //     = (τ(X) * Π(φᵢ(X)) * Σ 1/(φᵢ(X))) - Π(φᵢ(X)) * m(X)
      //     = Π(φᵢ(X)) * (τ(X) * Σ 1/(φᵢ(X)) - m(X))
      //     = Σᵢ(τ(X) * Π_{j != i} φⱼ(X)) - m(X) * Π(φᵢ(X))
      //
      // (1 - (l_last(X) + l_blind(X))) * (LHS - RHS) = 0
      size_t start = chunk_offset * chunk_size;
      for (size_t idx = 0; idx < chunk.size(); ++idx) {
        size_t cur_idx = start + idx;

        // φᵢ(X) = fᵢ(X) + β
        std::vector<F> inputs_value = base::Map(
            inputs_eval_data,
            [&inputs_evaluator, &cur_idx](
                size_t i, plonk::EvaluationInput<Evals>& input_eval_data) {
              return inputs_evaluator[i].Evaluate(input_eval_data, cur_idx,
                                                  /*scale=*/1, F::Zero());
            });

        // Π(φᵢ(X))
        F inputs_prod = std::accumulate(
            inputs_value.begin(), inputs_value.end(), F::One(),
            [](F& acc, const F& input) { return acc *= input; });

        // τ(X) = t(X) + β
        F table_value = table_evaluator.Evaluate(table_eval_data, cur_idx,
                                                 /*scale=*/1, F::Zero());
        RowIndex r_next =
            Rotation(1).GetIndex(cur_idx, /*scale=*/1, builder.n_);

        // LHS = τ(X) * Π(φᵢ(X)) * (ϕ(ω * X) - ϕ(X))
        F lhs = table_value * inputs_prod *
                (sum_coset[r_next] - sum_coset[cur_idx]);

        F inputs_exclusive = F::Zero();
        for (size_t i = 0; i < inputs_value.size(); ++i) {
          // Π_{j != i} φⱼ(X)
          F inputs_exclusive_prod = F::One();
          for (size_t j = 0; j < inputs_value.size(); ++j) {
            if (j != i) {
              inputs_exclusive_prod *= inputs_value[j];
            }
          }
          inputs_exclusive += inputs_exclusive_prod;
        }

        // RHS = Σᵢ τ(X) * Π_{j != i} φⱼ(X) - m(X) * Π(φᵢ(X))
        F rhs = table_value * inputs_exclusive - inputs_prod * m_coset[cur_idx];

        // l_first(X) * ϕ(X) = 0
        chunk[idx] *= builder.y_;
        chunk[idx] += builder.l_first_[cur_idx] * sum_coset[cur_idx];

        // l_last(X) * ϕ(X) = 0
        chunk[idx] *= builder.y_;
        chunk[idx] += builder.l_last_[cur_idx] * sum_coset[cur_idx];

        // (1 - (l_last(X) + l_blind(X))) * (lhs - rhs) = 0
        chunk[idx] *= builder.y_;
        chunk[idx] += (lhs - rhs) * builder.l_active_row_[cur_idx];
      }
    }
  }

  template <plonk::halo2::Vendor Vendor, typename PCS, typename LS>
  void UpdateLookupCosets(
      plonk::CircuitPolynomialBuilder<Vendor, PCS, LS>& builder,
      size_t circuit_idx) {
    using LookupProver = Prover<typename PCS::Poly, Evals>;

    size_t num_lookups =
        builder.lookup_provers_[circuit_idx].grand_sum_polys().size();
    const LookupProver& lookup_prover = builder.lookup_provers_[circuit_idx];
    lookup_sum_cosets_.resize(num_lookups);
    lookup_m_cosets_.resize(num_lookups);
    for (size_t i = 0; i < num_lookups; ++i) {
      lookup_sum_cosets_[i] =
          builder.coset_domain_->FFT(lookup_prover.grand_sum_polys()[i].poly());
      lookup_m_cosets_[i] =
          builder.coset_domain_->FFT(lookup_prover.m_polys()[i].poly());
    }
  }

 private:
  std::vector<LookupEvaluatorsPair<F>> lookup_evaluators_pairs_;
  std::vector<Evals> lookup_sum_cosets_;
  std::vector<Evals> lookup_m_cosets_;
};

}  // namespace tachyon::zk::lookup::log_derivative_halo2

#endif  // TACHYON_ZK_LOOKUP_LOG_DERIVATIVE_HALO2_EVALUATOR_H_
