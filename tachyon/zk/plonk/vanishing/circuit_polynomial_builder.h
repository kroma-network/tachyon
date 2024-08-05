// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_VANISHING_CIRCUIT_POLYNOMIAL_BUILDER_H_
#define TACHYON_ZK_PLONK_VANISHING_CIRCUIT_POLYNOMIAL_BUILDER_H_

#include <memory>
#include <memory_resource>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/base/containers/adapters.h"
#include "tachyon/base/parallelize.h"
#include "tachyon/base/types/always_false.h"
#include "tachyon/zk/base/rotation.h"
#include "tachyon/zk/lookup/prover.h"
#include "tachyon/zk/plonk/base/column_key.h"
#include "tachyon/zk/plonk/keys/proving_key_forward.h"
#include "tachyon/zk/plonk/permutation/permutation_prover.h"
#include "tachyon/zk/plonk/vanishing/evaluation_input.h"
#include "tachyon/zk/plonk/vanishing/graph_evaluator.h"
#include "tachyon/zk/plonk/vanishing/vanishing_utils.h"
#include "tachyon/zk/shuffle/prover.h"

namespace tachyon::zk {
namespace lookup::halo2 {

template <typename EvalsOrExtendedEvals>
class Evaluator;

}  // namespace lookup::halo2

namespace lookup::log_derivative_halo2 {

template <typename EvalsOrExtendedEvals>
class Evaluator;

}  // namespace lookup::log_derivative_halo2

namespace shuffle {

template <typename EvalsOrExtendedEvals>
class Evaluator;

}  // namespace shuffle

namespace plonk {

template <typename EvalsOrExtendedEvals>
class CustomGateEvaluator;

template <typename EvalsOrExtendedEvals>
class PermutationEvaluator;

// It generates "CircuitPolynomial" formed below:
// - gate₀(X) + y * gate₁(X) + ... + yⁱ * gateᵢ(X) + ...
// You can find more detailed theory in "Halo2 book"
// https://zcash.github.io/halo2/design/proving-system/vanishing.html
template <halo2::Vendor Vendor, typename PCS, typename LS>
class CircuitPolynomialBuilder {
 public:
  using F = typename PCS::Field;
  using C = typename PCS::Commitment;
  using Poly = typename PCS::Poly;
  using Evals = typename PCS::Evals;
  using Domain = typename PCS::Domain;
  using ExtendedDomain = typename PCS::ExtendedDomain;
  using ExtendedEvals = typename PCS::ExtendedEvals;
  using LookupProver = lookup::Prover<LS::kType, Poly, Evals>;

  using EvalsOrExtendedEvals =
      std::conditional_t<Vendor == halo2::Vendor::kPSE, ExtendedEvals, Evals>;

  CircuitPolynomialBuilder(
      const F& omega, const F& extended_omega, const F& theta, const F& beta,
      const F& gamma, const F& y, const F& zeta,
      const ProvingKey<Vendor, LS>& proving_key,
      const std::vector<PermutationProver<Poly, Evals>>& permutation_provers,
      const std::vector<LookupProver>& lookup_provers,
      const std::vector<shuffle::Prover<Poly, Evals>>& shuffle_provers,
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
        shuffle_provers_(shuffle_provers),
        poly_tables_(poly_tables) {}

  static CircuitPolynomialBuilder Create(
      const Domain* domain, const ExtendedDomain* extended_domain, size_t n,
      RowOffset last_row, size_t cs_degree,
      const std::vector<MultiPhaseRefTable<Poly>>& poly_tables, const F& theta,
      const F& beta, const F& gamma, const F& y, const F& zeta,
      const ProvingKey<Vendor, LS>& proving_key,
      const std::vector<PermutationProver<Poly, Evals>>& permutation_provers,
      const std::vector<LookupProver>& lookup_provers,
      const std::vector<shuffle::Prover<Poly, Evals>>& shuffle_provers) {
    CircuitPolynomialBuilder builder(
        domain->group_gen(), extended_domain->group_gen(), theta, beta, gamma,
        y, zeta, proving_key, permutation_provers, lookup_provers,
        shuffle_provers, poly_tables);
    builder.domain_ = domain;
    builder.extended_domain_ = extended_domain;

    builder.n_ = static_cast<int32_t>(n);
    builder.num_parts_ = extended_domain->size() >> domain->log_size_of_group();
    builder.chunk_len_ = cs_degree - 2;

    builder.delta_ = GetDelta<F>();

    builder.last_rotation_ = Rotation(last_row);
    builder.delta_start_ = beta * zeta;
    return builder;
  }

  // Returns an evaluation-formed polynomial as below.
  // - gate₀(X) + y * gate₁(X) + ... + yⁱ * gateᵢ(X) + ...
  ExtendedEvals BuildExtendedCircuitColumn(
      CustomGateEvaluator<Evals>& custom_gate_evaluator,
      PermutationEvaluator<Evals>& permutation_evaluator,
      lookup::Evaluator<LS::kType, Evals>& lookup_evaluator,
      shuffle::Evaluator<EvalsOrExtendedEvals>& shuffle_evaluator) {
    std::vector<std::vector<F>> value_parts;
    value_parts.reserve(num_parts_);
    // Calculate the quotient polynomial for each part
    for (size_t i = 0; i < num_parts_; ++i) {
      VLOG(1) << "BuildExtendedCircuitColumn part: (" << i + 1 << " / "
              << num_parts_ << ")";

      coset_domain_ = domain_->GetCoset(zeta_ * current_extended_omega_);

      UpdateLPolyCosets();

      std::vector<F> value_part(static_cast<size_t>(n_));
      size_t circuit_num = poly_tables_.size();
      for (size_t j = 0; j < circuit_num; ++j) {
        VLOG(1) << "BuildExtendedCircuitColumn part: " << i << " circuit: ("
                << j + 1 << " / " << circuit_num << ")";
        custom_gate_evaluator.UpdateCosets(*this, j);
        permutation_evaluator.UpdateCosets(*this, j);
        lookup_evaluator.UpdateCosets(*this, j);
        shuffle_evaluator.UpdateCosets(*this, j);

        base::Parallelize(
            value_part,
            [this, &custom_gate_evaluator, &permutation_evaluator,
             &lookup_evaluator, &shuffle_evaluator](
                absl::Span<F> chunk, size_t chunk_offset, size_t chunk_size) {
              custom_gate_evaluator.Evaluate(*this, chunk, chunk_offset,
                                             chunk_size);
              permutation_evaluator.Evaluate(*this, chunk, chunk_offset,
                                             chunk_size);
              lookup_evaluator.Evaluate(*this, chunk, chunk_offset, chunk_size);
              shuffle_evaluator.Evaluate(*this, chunk, chunk_offset,
                                         chunk_size);
            });
      }

      value_parts.push_back(std::move(value_part));
      current_extended_omega_ *= extended_omega_;
    }
    std::pmr::vector<F> extended = BuildExtendedColumnWithColumns(value_parts);
    return ExtendedEvals(std::move(extended));
  }

 private:
  friend class CustomGateEvaluator<EvalsOrExtendedEvals>;
  friend class PermutationEvaluator<EvalsOrExtendedEvals>;
  friend class lookup::halo2::Evaluator<EvalsOrExtendedEvals>;
  friend class lookup::log_derivative_halo2::Evaluator<EvalsOrExtendedEvals>;
  friend class shuffle::Evaluator<EvalsOrExtendedEvals>;

  EvaluationInput<EvalsOrExtendedEvals> ExtractEvaluationInput(
      std ::vector<F>&& intermediates, std::vector<int32_t>&& rotations) {
    return EvaluationInput<EvalsOrExtendedEvals>(std::move(intermediates),
                                                 std::move(rotations), table_,
                                                 theta_, beta_, gamma_, y_, n_);
  }

  void UpdateLPolyCosets() {
    l_first_ = coset_domain_->FFT(proving_key_.l_first());
    l_last_ = coset_domain_->FFT(proving_key_.l_last());
    l_active_row_ = coset_domain_->FFT(proving_key_.l_active_row());
  }

  // not owned
  const Domain* domain_ = nullptr;
  const ExtendedDomain* extended_domain_ = nullptr;
  std::unique_ptr<Domain> coset_domain_;

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

  const ProvingKey<Vendor, LS>& proving_key_;
  const std::vector<PermutationProver<Poly, Evals>>& permutation_provers_;
  const std::vector<LookupProver>& lookup_provers_;
  const std::vector<shuffle::Prover<Poly, Evals>>& shuffle_provers_;
  const std::vector<MultiPhaseRefTable<Poly>>& poly_tables_;

  EvalsOrExtendedEvals l_first_;
  EvalsOrExtendedEvals l_last_;
  EvalsOrExtendedEvals l_active_row_;

  MultiPhaseRefTable<EvalsOrExtendedEvals> table_;
};

}  // namespace plonk
}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_VANISHING_CIRCUIT_POLYNOMIAL_BUILDER_H_
