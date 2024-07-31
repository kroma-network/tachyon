// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_SHUFFLE_EVALUATOR_H_
#define TACHYON_ZK_SHUFFLE_EVALUATOR_H_

#include <memory>
#include <utility>
#include <vector>

#include "tachyon/zk/plonk/vanishing/circuit_polynomial_builder_forward.h"
#include "tachyon/zk/plonk/vanishing/graph_evaluator.h"
#include "tachyon/zk/plonk/vanishing/vanishing_utils.h"
#include "tachyon/zk/shuffle/prover.h"

namespace tachyon::zk::shuffle {

template <typename EvalsOrExtendedEvals>
class Evaluator {
 public:
  using F = typename EvalsOrExtendedEvals::Field;

  void Construct(const std::vector<Argument<F>>& shuffles) {
    // NOTE (chokobole): When constructing the graph, Scroll Halo2 uses beta,
    // whereas PSE Halo2 uses gamma. However, using beta here prevents the proof
    // from being verified. See
    // https://github.com/scroll-tech/halo2/blob/e5ddf67/halo2_proofs/src/plonk/evaluation.rs#L294-L307.
    shuffle_evaluators_.reserve(shuffles.size() * 2);
    for (const Argument<F>& shuffle : shuffles) {
      plonk::GraphEvaluator<F> graph_input;
      plonk::GraphEvaluator<F> graph_shuffle;

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

      // A_compressed(X) = θᵐ⁻¹A₀(X) + θᵐ⁻²A₁(X) + ... + θAₘ₋₂(X) + Aₘ₋₁(X)
      plonk::ValueSource compressed_input_coset =
          compress(graph_input, shuffle.input_expressions());
      // S_compressed(X) = θᵐ⁻¹S₀(X) + θᵐ⁻²S₁(X) + ... + θSₘ₋₂(X) + Sₘ₋₁(X)
      plonk::ValueSource compressed_shuffle_coset =
          compress(graph_shuffle, shuffle.shuffle_expressions());

      // A_compressed(X) + γ
      graph_input.AddCalculation(plonk::Calculation::Add(
          compressed_input_coset, plonk::ValueSource::Gamma()));
      // S_compressed(X) + γ
      graph_shuffle.AddCalculation(plonk::Calculation::Add(
          compressed_shuffle_coset, plonk::ValueSource::Gamma()));

      shuffle_evaluators_.push_back(std::move(graph_input));
      shuffle_evaluators_.push_back(std::move(graph_shuffle));
    }
  }

  template <plonk::halo2::Vendor Vendor, typename PCS, typename LS>
  void Evaluate(plonk::CircuitPolynomialBuilder<Vendor, PCS, LS>& builder,
                absl::Span<F> chunk, size_t chunk_offset, size_t chunk_size) {
    for (size_t i = 0; i < shuffle_evaluators_.size(); i += 2) {
      const plonk::GraphEvaluator<F>& input_evaluator = shuffle_evaluators_[i];
      const plonk::GraphEvaluator<F>& shuffle_evaluator =
          shuffle_evaluators_[i + 1];
      const EvalsOrExtendedEvals& product_coset = shuffle_product_cosets_[i];

      plonk::EvaluationInput<EvalsOrExtendedEvals> input_eval_data =
          builder.ExtractEvaluationInput(
              input_evaluator.CreateInitialIntermediates(),
              input_evaluator.CreateEmptyRotations());
      plonk::EvaluationInput<EvalsOrExtendedEvals> shuffle_eval_data =
          builder.ExtractEvaluationInput(
              shuffle_evaluator.CreateInitialIntermediates(),
              shuffle_evaluator.CreateEmptyRotations());

      size_t start = chunk_offset * chunk_size;
      for (size_t j = 0; j < chunk.size(); ++j) {
        size_t idx = start + j;

        F input_value = input_evaluator.Evaluate(input_eval_data, idx,
                                                 /*scale=*/1, F::Zero());
        F shuffle_value = shuffle_evaluator.Evaluate(shuffle_eval_data, idx,
                                                     /*scale=*/1, F::Zero());

        RowIndex r_next = Rotation(1).GetIndex(idx, /*scale=*/1, builder.n_);

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
        //  - B = z(ωX) * (θᵐ⁻¹ s₀(X) + ... + sₘ₋₁(X) + γ)
        //  - C = z(X) * (θᵐ⁻¹ a₀(X) + ... + aₘ₋₁(X) + γ)
        // clang-format on
        chunk[j] *= builder.y_;
        chunk[j] += builder.l_active_row_[idx] *
                    (product_coset[r_next] * shuffle_value -
                     product_coset[idx] * input_value);
      }
    }
  }

  template <plonk::halo2::Vendor Vendor, typename PCS, typename LS>
  void UpdateCosets(plonk::CircuitPolynomialBuilder<Vendor, PCS, LS>& builder,
                    size_t circuit_idx) {
    using Poly = typename PCS::Poly;
    using Evals = typename PCS::Evals;
    using ShuffleProver = Prover<Poly, Evals>;

    size_t num_shuffles =
        builder.shuffle_provers_[circuit_idx].grand_product_polys().size();
    const ShuffleProver& shuffle_prover = builder.shuffle_provers_[circuit_idx];
    shuffle_product_cosets_.resize(num_shuffles);

    for (size_t i = 0; i < num_shuffles; ++i) {
      if constexpr (Vendor == plonk::halo2::Vendor::kPSE) {
        shuffle_product_cosets_[i] = plonk::CoeffToExtended(
            shuffle_prover.grand_product_polys()[i].poly(),
            builder.extended_domain_);
      } else {
        shuffle_product_cosets_[i] = builder.coset_domain_->FFT(
            shuffle_prover.grand_product_polys()[i].poly());
      }
    }
  }

 private:
  std::vector<plonk::GraphEvaluator<F>> shuffle_evaluators_;
  std::vector<EvalsOrExtendedEvals> shuffle_product_cosets_;
};

}  // namespace tachyon::zk::shuffle

#endif  // TACHYON_ZK_SHUFFLE_EVALUATOR_H_
