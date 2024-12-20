// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_LOOKUP_HALO2_EVALUATOR_H_
#define TACHYON_ZK_LOOKUP_HALO2_EVALUATOR_H_

#include <memory>
#include <utility>
#include <vector>

#include "tachyon/zk/lookup/halo2/prover.h"
#include "tachyon/zk/plonk/vanishing/circuit_polynomial_builder_forward.h"
#include "tachyon/zk/plonk/vanishing/graph_evaluator.h"
#include "tachyon/zk/plonk/vanishing/vanishing_utils.h"

namespace tachyon::zk::lookup::halo2 {

template <typename EvalsOrExtendedEvals>
class Evaluator {
 public:
  using F = typename EvalsOrExtendedEvals::Field;

  void Construct(const std::vector<Argument<F>>& arguments) {
    TRACE_EVENT("ProofGeneration", "Lookup::Halo2::Evaluator::Construct");
    evaluators_.reserve(arguments.size());
    for (const Argument<F>& argument : arguments) {
      plonk::GraphEvaluator<F> graph;

      auto compress =
          [&graph](
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

      // A_compressed(X) = θᵐ⁻¹A₀(X) + θᵐ⁻²A₁(X) + ... + θAₘ₋₂(X) + Aₘ₋₁(X)
      plonk::ValueSource compressed_input_coset =
          compress(argument.input_expressions());
      // S_compressed(X) = θᵐ⁻¹S₀(X) + θᵐ⁻²S₁(X) + ... + θSₘ₋₂(X) + Sₘ₋₁(X)
      plonk::ValueSource compressed_table_coset =
          compress(argument.table_expressions());

      // A_compressed(X) + β
      plonk::ValueSource left = graph.AddCalculation(plonk::Calculation::Add(
          compressed_input_coset, plonk::ValueSource::Beta()));
      // S_compressed(X) + γ
      plonk::ValueSource right = graph.AddCalculation(plonk::Calculation::Add(
          compressed_table_coset, plonk::ValueSource::Gamma()));
      // (A_compressed(X) + β) * (S_compressed(X) + γ)
      graph.AddCalculation(plonk::Calculation::Mul(left, right));

      evaluators_.push_back(std::move(graph));
    }
  }

  template <typename PS>
  void Evaluate(plonk::CircuitPolynomialBuilder<PS>& builder,
                absl::Span<F> chunk, size_t chunk_offset, size_t chunk_size) {
    TRACE_EVENT("ProofGeneration", "Lookup::Halo2::Evaluator::Evaluate");
    for (size_t i = 0; i < evaluators_.size(); ++i) {
      const plonk::GraphEvaluator<F>& ev = evaluators_[i];
      const EvalsOrExtendedEvals& input_coset = input_cosets_[i];
      const EvalsOrExtendedEvals& table_coset = table_cosets_[i];
      const EvalsOrExtendedEvals& product_coset = product_cosets_[i];

      plonk::EvaluationInput<EvalsOrExtendedEvals> evaluation_input =
          builder.ExtractEvaluationInput(ev.CreateInitialIntermediates(),
                                         ev.CreateEmptyRotations());

      size_t start = chunk_offset * chunk_size;
      for (size_t j = 0; j < chunk.size(); ++j) {
        size_t idx = start + j;

        F zero = F::Zero();
        F table_value = ev.Evaluate(evaluation_input, idx, /*scale=*/1, zero);

        RowIndex r_next = Rotation(1).GetIndex(idx, /*scale=*/1, builder.n_);
        RowIndex r_prev = Rotation(-1).GetIndex(idx, /*scale=*/1, builder.n_);

        F a_minus_s = input_coset[idx] - table_coset[idx];

        // l_first(X) * (1 - z(X)) = 0
        chunk[j] *= builder.y_;
        chunk[j] += builder.l_first_[idx] * (F::One() - product_coset[idx]);

        // l_last(X) * (z(X)² - z(X)) = 0
        chunk[j] *= builder.y_;
        chunk[j] += builder.l_last_[idx] *
                    (product_coset[idx].Square() - product_coset[idx]);

        // clang-format off
        // A * (B - C) = 0 where
        //  - A = 1 - (l_last(X) + l_blind(X))
        //  - B = z(ωX) * (a'(X) + β) * (s'(X) + γ)
        //  - C = z(X) * (θᵐ⁻¹ a₀(X) + ... + aₘ₋₁(X) + β) * (θᵐ⁻¹ s₀(X) + ... + sₘ₋₁(X) + γ)
        // clang-format on
        chunk[j] *= builder.y_;
        chunk[j] +=
            builder.l_active_row_[idx] *
            (product_coset[r_next] * (input_coset[idx] + builder.beta_) *
                 (table_coset[idx] + builder.gamma_) -
             product_coset[idx] * table_value);

        // Check that the first values in the permuted input expression and
        // permuted fixed expression are the same.
        // l_first(X) * (a'(X) - s'(X)) = 0
        chunk[j] *= builder.y_;
        chunk[j] += builder.l_first_[idx] * a_minus_s;

        // Check that each value in the permuted lookup input expression is
        // either equal to the value above it, or the value at the same
        // index in the permuted table expression. (1 - (l_last + l_blind)) *
        // (a′(X) − s′(X)) * (a′(X) − a′(ω⁻¹X)) = 0
        chunk[j] *= builder.y_;
        chunk[j] += builder.l_active_row_[idx] * a_minus_s *
                    (input_coset[idx] - input_coset[r_prev]);
      }
    }
  }

  template <typename PS>
  void UpdateCosets(plonk::CircuitPolynomialBuilder<PS>& builder,
                    size_t circuit_idx) {
    TRACE_EVENT("ProofGeneration", "Lookup::Halo2::Evaluator::UpdateCosets");
    using PCS = typename PS::PCS;
    using Poly = typename PCS::Poly;
    using Evals = typename PCS::Evals;
    using LookupProver = Prover<Poly, Evals>;

    size_t num_lookups =
        builder.lookup_provers_[circuit_idx].grand_product_polys().size();
    if (num_lookups == 0) return;

    const LookupProver& prover = builder.lookup_provers_[circuit_idx];
    product_cosets_.resize(num_lookups);
    input_cosets_.resize(num_lookups);
    table_cosets_.resize(num_lookups);
    for (size_t i = 0; i < num_lookups; ++i) {
      if constexpr (PS::kVendor == plonk::halo2::Vendor::kPSE) {
        product_cosets_[i] = plonk::CoeffToExtended(
            prover.grand_product_polys()[i].poly(), builder.extended_domain_);
        input_cosets_[i] =
            plonk::CoeffToExtended(prover.permuted_pairs()[i].input().poly(),
                                   builder.extended_domain_);
        table_cosets_[i] =
            plonk::CoeffToExtended(prover.permuted_pairs()[i].table().poly(),
                                   builder.extended_domain_);
      } else {
        product_cosets_[i] =
            builder.coset_domain_->FFT(prover.grand_product_polys()[i].poly());
        input_cosets_[i] = builder.coset_domain_->FFT(
            prover.permuted_pairs()[i].input().poly());
        table_cosets_[i] = builder.coset_domain_->FFT(
            prover.permuted_pairs()[i].table().poly());
      }
    }
  }

 private:
  std::vector<plonk::GraphEvaluator<F>> evaluators_;
  std::vector<EvalsOrExtendedEvals> product_cosets_;
  std::vector<EvalsOrExtendedEvals> input_cosets_;
  std::vector<EvalsOrExtendedEvals> table_cosets_;
};

}  // namespace tachyon::zk::lookup::halo2

#endif  // TACHYON_ZK_LOOKUP_HALO2_EVALUATOR_H_
