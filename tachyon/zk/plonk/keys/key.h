// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_KEYS_KEY_H_
#define TACHYON_ZK_PLONK_KEYS_KEY_H_

#include <utility>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/zk/base/entities/entity.h"
#include "tachyon/zk/plonk/constraint_system.h"
#include "tachyon/zk/plonk/keys/assembly.h"

namespace tachyon::zk {

template <typename PCS>
class Key {
 public:
  using F = typename PCS::Field;
  using Domain = typename PCS::Domain;
  using Evals = typename PCS::Evals;

  static Assembly<PCS> CreateAssembly(
      const Domain* domain, const ConstraintSystem<F>& constraint_system) {
    using RationalEvals = typename Assembly<PCS>::RationalEvals;
    size_t n = domain->size();
    return {
        base::CreateVector(constraint_system.num_fixed_columns(),
                           domain->template Empty<RationalEvals>()),
        PermutationAssembly<PCS>(constraint_system.permutation(), n),
        base::CreateVector(constraint_system.num_selectors(),
                           base::CreateVector(n, false)),
        // NOTE(chokobole): Considering that this is called from a verifier,
        // then you can't load this number through |prover->GetUsableRows()|.
        base::Range<size_t>::Until(
            n - (constraint_system.ComputeBlindingFactors() + 1))};
  }

 protected:
  struct PreLoadResult {
    ConstraintSystem<F> constraint_system;
    Assembly<PCS> assembly;
    std::vector<Evals> fixed_columns;
  };

  template <typename Circuit>
  bool PreLoad(Entity<PCS>* entity, const Circuit& circuit,
               PreLoadResult* result) {
    using Config = typename Circuit::Config;
    using FloorPlanner = typename Circuit::FloorPlanner;
    using RationalEvals = typename Assembly<PCS>::RationalEvals;
    using ExtendedDomain = typename PCS::ExtendedDomain;

    ConstraintSystem<F>& constraint_system = result->constraint_system;
    Config config = Circuit::Configure(constraint_system);

    PCS& pcs = entity->pcs();
    if (pcs.N() < constraint_system.ComputeMinimumRows()) {
      LOG(ERROR) << "Not enough rows available " << pcs.N() << " vs "
                 << constraint_system.ComputeMinimumRows();
      return false;
    }
    size_t extended_k = constraint_system.ComputeExtendedDegree(pcs.K());
    entity->set_extended_domain(
        ExtendedDomain::Create(size_t{1} << extended_k));

    result->assembly = CreateAssembly(entity->domain(), constraint_system);
    Assembly<PCS>& assembly = result->assembly;
    FloorPlanner::Synthesize(&assembly, circuit, std::move(config),
                             constraint_system.constants());

    result->fixed_columns =
        base::Map(assembly.fixed_columns(), [](const RationalEvals& evals) {
          std::vector<F> result =
              base::CreateVector(evals.evaluations().size(), F::Zero());
          CHECK(math::RationalField<F>::BatchEvaluate(evals.evaluations(),
                                                      &result));
          return Evals(std::move(result));
        });
    std::vector<Evals>& fixed_columns = result->fixed_columns;

    std::vector<std::vector<F>> selector_polys_tmp =
        constraint_system.CompressSelectors(assembly.selectors());
    std::vector<Evals> selector_polys =
        base::Map(std::make_move_iterator(selector_polys_tmp.begin()),
                  std::make_move_iterator(selector_polys_tmp.end()),
                  [](std::vector<F>&& vec) { return Evals(std::move(vec)); });
    fixed_columns.insert(fixed_columns.end(),
                         std::make_move_iterator(selector_polys.begin()),
                         std::make_move_iterator(selector_polys.end()));

    return true;
  }
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_KEYS_KEY_H_
