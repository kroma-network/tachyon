// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_KEYS_KEY_H_
#define TACHYON_ZK_PLONK_KEYS_KEY_H_

#include <stddef.h>
#include <stdint.h>

#include <utility>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/export.h"
#include "tachyon/zk/base/entities/entity.h"
#include "tachyon/zk/plonk/constraint_system/constraint_system.h"
#include "tachyon/zk/plonk/keys/assembly.h"

namespace tachyon::zk::plonk {

template <typename Evals, typename RationalEvals>
struct KeyPreLoadResult {
  using F = typename Evals::Field;

  ConstraintSystem<F> constraint_system;
  Assembly<RationalEvals> assembly;
  std::vector<Evals> fixed_columns;

  KeyPreLoadResult() = default;
  explicit KeyPreLoadResult(lookup::Type lookup_type) {
    constraint_system.set_lookup_type(lookup_type);
  }
};

class TACHYON_EXPORT Key {
 public:
  template <typename RationalEvals, typename Domain, typename F>
  static Assembly<RationalEvals> CreateAssembly(
      const Domain* domain, const ConstraintSystem<F>& constraint_system) {
    // NOTE(chokobole): It's safe to downcast because domain is already checked.
    RowIndex n = static_cast<RowIndex>(domain->size());
    return {
        std::vector<RationalEvals>(constraint_system.num_fixed_columns(),
                                   domain->template Zero<RationalEvals>()),
        PermutationAssembly(constraint_system.permutation(), n),
        std::vector<std::vector<bool>>(constraint_system.GetNumSelectors(),
                                       std::vector<bool>(n, false)),
        // NOTE(chokobole): Considering that this is called from a verifier,
        // then you can't load this number through |prover->GetUsableRows()|.
        base::Range<RowIndex>::Until(
            n - (constraint_system.ComputeBlindingFactors() + 1))};
  }

 protected:
  template <typename PCS, typename Circuit, typename Evals,
            typename RationalEvals>
  bool PreLoad(Entity<PCS>* entity, const Circuit& circuit,
               KeyPreLoadResult<Evals, RationalEvals>* result) {
    using F = typename Evals::Field;
    using Config = typename Circuit::Config;
    using FloorPlanner = typename Circuit::FloorPlanner;
    using ExtendedDomain = typename PCS::ExtendedDomain;

    ConstraintSystem<F>& constraint_system = result->constraint_system;
    Config config = Circuit::Configure(constraint_system);

    if (constraint_system.lookup_type() == lookup::Type::kLogDerivativeHalo2) {
      constraint_system.ChunkLookups();
    }

    PCS& pcs = entity->pcs();
    if (pcs.N() < constraint_system.ComputeMinimumRows()) {
      LOG(ERROR) << "Not enough rows available " << pcs.N() << " vs "
                 << constraint_system.ComputeMinimumRows();
      return false;
    }
    uint32_t extended_k = constraint_system.ComputeExtendedK(pcs.K());
    entity->set_extended_domain(
        ExtendedDomain::Create(size_t{1} << extended_k));

    result->assembly =
        CreateAssembly<RationalEvals>(entity->domain(), constraint_system);
    Assembly<RationalEvals>& assembly = result->assembly;
    FloorPlanner floor_planner;
    floor_planner.Synthesize(&assembly, circuit, std::move(config),
                             constraint_system.constants());

    result->fixed_columns =
        base::Map(assembly.fixed_columns(), [](const RationalEvals& evals) {
          std::vector<F> result(evals.evaluations().size());
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

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_KEYS_KEY_H_
