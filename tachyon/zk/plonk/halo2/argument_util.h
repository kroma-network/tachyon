#ifndef TACHYON_ZK_PLONK_HALO2_ARGUMENT_UTIL_H_
#define TACHYON_ZK_PLONK_HALO2_ARGUMENT_UTIL_H_

#include <stddef.h>

#include <utility>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/numerics/checked_math.h"
#include "tachyon/base/ref.h"
#include "tachyon/zk/base/entities/prover_base.h"
#include "tachyon/zk/base/point_set.h"
#include "tachyon/zk/lookup/lookup_argument_runner.h"
#include "tachyon/zk/plonk/base/ref_table.h"
#include "tachyon/zk/plonk/constraint_system/constraint_system.h"
#include "tachyon/zk/plonk/constraint_system/query.h"
#include "tachyon/zk/plonk/halo2/step_returns.h"
#include "tachyon/zk/plonk/keys/proving_key.h"
#include "tachyon/zk/plonk/permutation/permutation_argument_runner.h"

namespace tachyon::zk::halo2 {

template <typename PCS, typename F, typename Evals,
          typename Poly = typename PCS::Poly>
std::vector<std::vector<LookupPermuted<Poly, Evals>>> BatchPermuteLookups(
    ProverBase<PCS>* prover,
    const std::vector<LookupArgument<F>>& lookup_arguments,
    const std::vector<RefTable<Evals>>& tables, absl::Span<const F> challenges,
    const F& theta) {
  size_t num_circuits = tables.size();
  base::CheckedNumeric<int32_t> n_tmp = prover->pcs().N();
  int32_t n = n_tmp.ValueOrDie();
  return base::CreateVector(num_circuits, [prover, challenges,
                                           &lookup_arguments, &tables, &theta,
                                           n](size_t i) {
    const RefTable<Evals>& table = tables[i];
    return base::Map(
        lookup_arguments, [prover, challenges, &table, &theta,
                           n](const LookupArgument<F>& lookup_argument) {
          SimpleEvaluator<Evals> simple_evaluator(0, n, 1, table, challenges);
          return LookupArgumentRunner<Poly, Evals>::PermuteArgument(
              prover, lookup_argument, theta, simple_evaluator);
        });
  });
}

template <typename PCS, typename Poly, typename Evals, typename F>
std::vector<std::vector<LookupCommitted<Poly>>> BatchCommitLookups(
    ProverBase<PCS>* prover,
    std::vector<std::vector<LookupPermuted<Poly, Evals>>>&&
        permuted_lookups_vec,
    const F& beta, const F& gamma) {
  return base::Map(
      permuted_lookups_vec,
      [prover, &beta,
       &gamma](std::vector<LookupPermuted<Poly, Evals>>& permuted_lookups) {
        return base::Map(
            permuted_lookups,
            [prover, &beta,
             &gamma](LookupPermuted<Poly, Evals>& permuted_lookup) {
              return LookupArgumentRunner<Poly, Evals>::CommitPermuted(
                  prover, std::move(permuted_lookup), beta, gamma);
            });
      });
}

template <typename PCS, typename Poly, typename F>
std::vector<std::vector<LookupEvaluated<Poly>>> BatchEvaluateLookups(
    ProverBase<PCS>* prover,
    std::vector<std::vector<LookupCommitted<Poly>>>&& committed_lookups_vec,
    const F& x) {
  return base::Map(
      committed_lookups_vec,
      [prover, &x](std::vector<LookupCommitted<Poly>>& committed_lookups) {
        return base::Map(
            committed_lookups,
            [prover, &x](LookupCommitted<Poly>& committed_lookup) {
              return LookupArgumentRunner<Poly, typename PCS::Evals>::
                  EvaluateCommitted(prover, std::move(committed_lookup), x);
            });
      });
}

template <typename PCS, typename Poly, typename Evals, typename F>
std::vector<PermutationCommitted<Poly>> BatchCommitPermutations(
    ProverBase<PCS>* prover, const PermutationArgument& permutation_argument,
    const PermutationProvingKey<Poly, Evals>& permutation_proving_key,
    const std::vector<RefTable<Evals>>& tables, size_t cs_degree, const F& beta,
    const F& gamma) {
  size_t num_circuits = tables.size();
  return base::CreateVector(num_circuits, [&](size_t i) {
    return PermutationArgumentRunner<Poly, Evals>::CommitArgument(
        prover, permutation_argument, tables[i], cs_degree,
        permutation_proving_key, beta, gamma);
  });
}

template <typename PCS, typename Poly, typename F>
std::vector<PermutationEvaluated<Poly>> BatchEvaluatePermutations(
    ProverBase<PCS>* prover,
    std::vector<PermutationCommitted<Poly>>&& committed_permutations,
    const F& x) {
  return base::Map(
      committed_permutations,
      [&](PermutationCommitted<Poly>& committed_permutation) {
        return PermutationArgumentRunner<Poly, typename PCS::Evals>::
            EvaluateCommitted(prover, std::move(committed_permutation), x);
      });
}

template <typename PCS, typename Poly, typename F, ColumnType C>
void EvaluatePolysByQueries(ProverBase<PCS>* prover,
                            const absl::Span<const Poly> polys,
                            const std::vector<QueryData<C>>& queries,
                            const F& x) {
  for (const QueryData<C>& query : queries) {
    const Poly& poly = polys[query.column().index()];
    prover->EvaluateAndWriteToProof(
        poly, query.rotation().RotateOmega(prover->domain(), x));
  }
}

template <typename PCS, typename Poly, typename F>
void EvaluateColumns(ProverBase<PCS>* prover,
                     const ConstraintSystem<F>& constraint_system,
                     const std::vector<RefTable<Poly>>& tables, const F& x) {
  size_t num_circuits = tables.size();
  if constexpr (PCS::kQueryInstance) {
    for (size_t i = 0; i < num_circuits; ++i) {
      EvaluatePolysByQueries(prover, tables[i].GetInstanceColumns(),
                             constraint_system.instance_queries(), x);
    }
  }
  for (size_t i = 0; i < num_circuits; ++i) {
    EvaluatePolysByQueries(prover, tables[i].GetAdviceColumns(),
                           constraint_system.advice_queries(), x);
  }

  EvaluatePolysByQueries(prover, tables[0].GetFixedColumns(),
                         constraint_system.fixed_queries(), x);
}

template <typename PCS, typename P, typename L, typename V>
size_t GetNumOpenings(const ProvingKey<PCS>& proving_key,
                      const StepReturns<P, L, V>& evaluated,
                      size_t num_circuits) {
  const ConstraintSystem<typename PCS::Field>& cs =
      proving_key.verifying_key().constraint_system();

  base::CheckedNumeric<size_t> num_openings = cs.advice_queries().size();
  num_openings += cs.fixed_queries().size();
  if constexpr (PCS::kQueryInstance) {
    num_openings += cs.instance_queries().size();
  }
  for (size_t i = 0; i < num_circuits; ++i) {
    if (!evaluated.permutations().empty()) {
      num_openings += evaluated.permutations()[i].product_polys().size();
    }
    if (!evaluated.lookups_vec().empty()) {
      num_openings += evaluated.lookups_vec()[i].size();
    }
  }
  num_openings += proving_key.permutation_proving_key().permutations().size();

  // Ensure that there is exactly one opening for vanishing.
  return (num_openings + 1).ValueOrDie();
}

template <typename PCS, typename Poly, typename F, ColumnType C>
std::vector<crypto::PolynomialOpening<Poly>> GenerateColumnOpenings(
    const ProverBase<PCS>* prover, absl::Span<const Poly> polys,
    const std::vector<QueryData<C>>& queries, const F& x, PointSet<F>& points) {
  return base::Map(
      queries, [prover, polys, &points, &x](const QueryData<C>& query) mutable {
        const F point = query.rotation().RotateOmega(prover->domain(), x);
        base::DeepRef<const F> point_ref = points.Insert(point);
        const Poly& poly = polys[query.column().index()];
        return crypto::PolynomialOpening<Poly>(base::DeepRef<const Poly>(&poly),
                                               point_ref, poly.Evaluate(point));
      });
}

}  // namespace tachyon::zk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_ARGUMENT_UTIL_H_
