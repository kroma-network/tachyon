// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_VANISHING_VANISHING_ARGUMENT_H_
#define TACHYON_ZK_PLONK_VANISHING_VANISHING_ARGUMENT_H_

#include <stdint.h>

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "tachyon/zk/base/entities/prover_base.h"
#include "tachyon/zk/lookup/evaluator.h"
#include "tachyon/zk/plonk/constraint_system/constraint_system.h"
#include "tachyon/zk/plonk/keys/proving_key_forward.h"
#include "tachyon/zk/plonk/permutation/permutation_evaluator.h"
#include "tachyon/zk/plonk/vanishing/circuit_polynomial_builder.h"
#include "tachyon/zk/plonk/vanishing/custom_gate_evaluator.h"
#include "tachyon/zk/shuffle/evaluator.h"

namespace tachyon::zk::plonk {

template <halo2::Vendor Vendor, typename LS>
class VanishingArgument {
 public:
  using F = typename LS::Field;
  using Poly = typename LS::Poly;
  using Evals = typename LS::Evals;
  using ExtendedEvals = typename LS::ExtendedEvals;
  using LookupProver = lookup::Prover<LS::kType, Poly, Evals>;

  using EvalsOrExtendedEvals =
      std::conditional_t<Vendor == halo2::Vendor::kPSE, ExtendedEvals, Evals>;

  VanishingArgument() = default;

  static VanishingArgument Create(
      const ConstraintSystem<F>& constraint_system) {
    VanishingArgument evaluator;
    evaluator.custom_gate_evaluator_.Construct(constraint_system.gates());
    evaluator.lookup_evaluator_.Construct(constraint_system.lookups());
    evaluator.shuffle_evaluator_.Construct(constraint_system.shuffles());
    return evaluator;
  }

  template <typename PCS>
  ExtendedEvals BuildExtendedCircuitColumn(
      ProverBase<PCS>* prover, const ProvingKey<Vendor, LS>& proving_key,
      const std::vector<MultiPhaseRefTable<Poly>>& poly_tables, const F& theta,
      const F& beta, const F& gamma, const F& y, const F& zeta,
      const std::vector<PermutationProver<Poly, Evals>>& permutation_provers,
      const std::vector<LookupProver>& lookup_provers,
      const std::vector<shuffle::Prover<Poly, Evals>>& shuffle_provers) {
    size_t cs_degree =
        proving_key.verifying_key().constraint_system().ComputeDegree();

    CircuitPolynomialBuilder<Vendor, PCS, LS> builder =
        CircuitPolynomialBuilder<Vendor, PCS, LS>::Create(
            prover->domain(), prover->extended_domain(), prover->pcs().N(),
            prover->GetLastRow(), cs_degree, poly_tables, theta, beta, gamma, y,
            zeta, proving_key, permutation_provers, lookup_provers,
            shuffle_provers);

    return builder.BuildExtendedCircuitColumn(
        custom_gate_evaluator_, permutation_evaluator_, lookup_evaluator_,
        shuffle_evaluator_);
  }

 private:
  CustomGateEvaluator<EvalsOrExtendedEvals> custom_gate_evaluator_;
  PermutationEvaluator<EvalsOrExtendedEvals> permutation_evaluator_;
  lookup::Evaluator<LS::kType, EvalsOrExtendedEvals> lookup_evaluator_;
  shuffle::Evaluator<EvalsOrExtendedEvals> shuffle_evaluator_;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_VANISHING_VANISHING_ARGUMENT_H_
