#ifndef TACHYON_ZK_PLONK_HALO2_ARGUMENT_H_
#define TACHYON_ZK_PLONK_HALO2_ARGUMENT_H_

#include <utility>
#include <vector>

#include "tachyon/base/logging.h"
#include "tachyon/crypto/commitments/polynomial_openings.h"
#include "tachyon/zk/plonk/halo2/argument_data.h"
#include "tachyon/zk/plonk/halo2/argument_util.h"
#include "tachyon/zk/plonk/vanishing/prover_vanishing_argument.h"
#include "tachyon/zk/plonk/vanishing/vanishing_argument.h"

namespace tachyon::zk::halo2 {

// Data class including all arguments for creating proof.
template <typename PCS>
class Argument {
 public:
  using F = typename PCS::Field;
  using Poly = typename PCS::Poly;
  using Evals = typename PCS::Evals;
  using Domain = typename PCS::Domain;
  using ExtendedEvals = typename PCS::ExtendedEvals;

  Argument() = default;

  // NOTE(chokobole): This is used by rust halo2 binding.
  Argument(const std::vector<Evals>* fixed_columns,
           const std::vector<Poly>* fixed_polys,
           ArgumentData<PCS>* argument_data)
      : fixed_columns_(fixed_columns),
        fixed_polys_(fixed_polys),
        argument_data_(argument_data) {}

  void TransformAdvice(const Domain* domain) {
    return argument_data_->TransformAdvice(domain);
  }

  std::vector<std::vector<LookupPermuted<Poly, Evals>>> CompressLookupStep(
      ProverBase<PCS>* prover, const ConstraintSystem<F>& constraint_system,
      const F& theta) const {
    std::vector<RefTable<Evals>> tables = argument_data_->ExportColumnTables(
        absl::MakeConstSpan(*fixed_columns_));
    return BatchPermuteLookups(prover, constraint_system.lookups(), tables,
                               argument_data_->GetChallenges(), theta);
  }

  StepReturns<PermutationCommitted<Poly>, LookupCommitted<Poly>,
              VanishingCommitted<PCS>>
  CommitCircuitStep(
      ProverBase<PCS>* prover, const ConstraintSystem<F>& constraint_system,
      const PermutationProvingKey<Poly, Evals>& permutation_proving_key,
      std::vector<std::vector<LookupPermuted<Poly, Evals>>>&&
          permuted_lookups_vec,
      const F& beta, const F& gamma) {
    std::vector<RefTable<Evals>> tables = argument_data_->ExportColumnTables(
        absl::MakeConstSpan(*fixed_columns_));

    std::vector<PermutationCommitted<Poly>> committed_permutations =
        BatchCommitPermutations(prover, constraint_system.permutation(),
                                permutation_proving_key, tables,
                                constraint_system.ComputeDegree(), beta, gamma);

    std::vector<std::vector<LookupCommitted<Poly>>> committed_lookups_vec =
        BatchCommitLookups(prover, std::move(permuted_lookups_vec), beta,
                           gamma);

    VanishingCommitted<PCS> vanishing_committed;
    CHECK(CommitRandomPoly(prover, &vanishing_committed));

    return {std::move(committed_permutations), std::move(committed_lookups_vec),
            std::move(vanishing_committed)};
  }

  template <typename P, typename L, typename V>
  ExtendedEvals GenerateCircuitPolynomial(ProverBase<PCS>* prover,
                                          const ProvingKey<PCS>& proving_key,
                                          const StepReturns<P, L, V>& committed,
                                          const F& beta, const F& gamma,
                                          const F& theta, const F& y) const {
    VanishingArgument<F> vanishing_argument = VanishingArgument<F>::Create(
        proving_key.verifying_key().constraint_system());
    F zeta = GetHalo2Zeta<F>();
    return vanishing_argument.BuildExtendedCircuitColumn(
        prover, proving_key, beta, gamma, theta, y, zeta,
        argument_data_->GetChallenges(), committed.permutations(),
        committed.lookups_vec(),
        argument_data_->ExportPolyTables(absl::MakeConstSpan(*fixed_polys_)));
  }

  template <typename P, typename L, typename V>
  StepReturns<PermutationEvaluated<Poly>, LookupEvaluated<Poly>,
              VanishingEvaluated<PCS>>
  EvaluateCircuitStep(ProverBase<PCS>* prover,
                      const ProvingKey<PCS>& proving_key,
                      StepReturns<P, L, V>& committed,
                      VanishingConstructed<PCS>&& constructed_vanishing,
                      const F& x) const {
    const ConstraintSystem<F>& cs =
        proving_key.verifying_key().constraint_system();
    std::vector<RefTable<Poly>> tables =
        argument_data_->ExportPolyTables(absl::MakeConstSpan(*fixed_polys_));
    EvaluateColumns(prover, cs, tables, x);

    F xn = x.Pow(prover->pcs().N());
    VanishingEvaluated<PCS> evaluated_vanishing;
    CHECK(CommitRandomEval(prover->pcs(), std::move(constructed_vanishing), x,
                           xn, prover->GetWriter(), &evaluated_vanishing));

    PermutationArgumentRunner<Poly, Evals>::EvaluateProvingKey(
        prover, proving_key.permutation_proving_key(), x);

    std::vector<PermutationEvaluated<Poly>> evaluated_permutations =
        BatchEvaluatePermutations(prover,
                                  std::move(committed).TakePermutations(), x);

    std::vector<std::vector<LookupEvaluated<Poly>>> evaluated_lookups_vec =
        BatchEvaluateLookups(prover, std::move(committed).TakeLookupsVec(), x);

    return {std::move(evaluated_permutations), std::move(evaluated_lookups_vec),
            std::move(evaluated_vanishing)};
  }

  template <typename P, typename L, typename V>
  std::vector<crypto::PolynomialOpening<Poly>> ConstructOpenings(
      ProverBase<PCS>* prover, const ProvingKey<PCS>& proving_key,
      const StepReturns<P, L, V>& evaluated, const F& x) {
    std::vector<RefTable<Poly>> tables =
        argument_data_->ExportPolyTables(absl::MakeConstSpan(*fixed_polys_));
    const ConstraintSystem<F>& cs =
        proving_key.verifying_key().constraint_system();
    opening_points_set_.Insert(x);

    size_t num_circuits = argument_data_->GetNumCircuits();
    std::vector<crypto::PolynomialOpening<Poly>> ret;
    ret.reserve(GetNumOpenings(proving_key, evaluated, num_circuits));
    for (size_t i = 0; i < num_circuits; ++i) {
      // Generate openings for instances columns of the specific circuit.
      if constexpr (PCS::kQueryInstance) {
        std::vector<crypto::PolynomialOpening<Poly>> openings =
            GenerateColumnOpenings(prover, tables[i].GetInstanceColumns(),
                                   cs.instance_queries(), x,
                                   opening_points_set_);
        ret.insert(ret.end(), std::make_move_iterator(openings.begin()),
                   std::make_move_iterator(openings.end()));
      }

      // Generate openings for advices columns of the specific circuit.
      std::vector<crypto::PolynomialOpening<Poly>> openings =
          GenerateColumnOpenings(prover, tables[i].GetAdviceColumns(),
                                 cs.advice_queries(), x, opening_points_set_);
      ret.insert(ret.end(), std::make_move_iterator(openings.begin()),
                 std::make_move_iterator(openings.end()));

      // Generate openings for permutation columns of the specific circuit.
      openings = PermutationArgumentRunner<Poly, Evals>::OpenEvaluated(
          prover, evaluated.permutations()[i], x, opening_points_set_);
      ret.insert(ret.end(), std::make_move_iterator(openings.begin()),
                 std::make_move_iterator(openings.end()));

      // Generate openings for lookup columns of the specific circuit.
      openings = base::FlatMap(
          evaluated.lookups_vec()[i],
          [prover, &x, this](const LookupEvaluated<Poly>& evaluated_lookup) {
            return LookupArgumentRunner<Poly, Evals>::OpenEvaluated(
                prover, evaluated_lookup, x, opening_points_set_);
          });
      ret.insert(ret.end(), std::make_move_iterator(openings.begin()),
                 std::make_move_iterator(openings.end()));
    }

    // Generate openings for fixed columns.
    // NOTE(dongchangYoo): |fixed_xx|s of each |tables[i]| are equal each other.
    std::vector<crypto::PolynomialOpening<Poly>> openings =
        GenerateColumnOpenings(prover, tables[0].GetFixedColumns(),
                               cs.fixed_queries(), x, opening_points_set_);
    ret.insert(ret.end(), std::make_move_iterator(openings.begin()),
               std::make_move_iterator(openings.end()));

    openings =
        PermutationArgumentRunner<Poly, Evals>::OpenPermutationProvingKey(
            proving_key.permutation_proving_key(), x);
    ret.insert(ret.end(), std::make_move_iterator(openings.begin()),
               std::make_move_iterator(openings.end()));

    openings = OpenVanishingArgument(evaluated.vanishing(), x);
    ret.insert(ret.end(), std::make_move_iterator(openings.begin()),
               std::make_move_iterator(openings.end()));

    return ret;
  }

 private:
  // not owned
  const std::vector<Evals>* fixed_columns_ = nullptr;
  // not owned
  const std::vector<Poly>* fixed_polys_ = nullptr;
  // not owned
  ArgumentData<PCS>* argument_data_ = nullptr;

  // NOTE(dongchangYoo): set of points which will be included to any openings.
  PointSet<F> opening_points_set_;
};

}  // namespace tachyon::zk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_ARGUMENT_H_
