#ifndef TACHYON_ZK_PLONK_HALO2_ARGUMENT_H_
#define TACHYON_ZK_PLONK_HALO2_ARGUMENT_H_

#include <utility>
#include <vector>

#include "tachyon/base/logging.h"
#include "tachyon/crypto/commitments/polynomial_openings.h"
#include "tachyon/zk/plonk/halo2/argument_util.h"
#include "tachyon/zk/plonk/halo2/synthesizer.h"
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
  Argument(size_t num_circuits, const std::vector<Evals>* fixed_columns,
           const std::vector<Poly>* fixed_polys,
           std::vector<std::vector<Evals>>&& advice_columns_vec,
           std::vector<std::vector<F>>&& advice_blinds_vec,
           std::vector<F>&& challenges,
           std::vector<std::vector<Evals>>&& instance_columns_vec,
           std::vector<std::vector<Poly>>&& instance_polys_vec)
      : num_circuits_(num_circuits),
        fixed_columns_(fixed_columns),
        fixed_polys_(fixed_polys),
        advice_columns_vec_(std::move(advice_columns_vec)),
        advice_blinds_vec_(std::move(advice_blinds_vec)),
        challenges_(std::move(challenges)),
        instance_columns_vec_(std::move(instance_columns_vec)),
        instance_polys_vec_(std::move(instance_polys_vec)) {
    CHECK_EQ(num_circuits_, advice_columns_vec_.size());
    CHECK_EQ(num_circuits_, advice_blinds_vec_.size());
    CHECK_EQ(num_circuits_, instance_columns_vec_.size());
    CHECK_EQ(num_circuits_, instance_polys_vec_.size());
  }

  template <typename Circuit>
  static Argument Create(
      ProverBase<PCS>* prover, std::vector<Circuit>& circuits,
      const std::vector<Evals>* fixed_columns,
      const std::vector<Poly>* fixed_polys,
      const ConstraintSystem<F>& constraint_system,
      std::vector<std::vector<Evals>>&& instance_columns_vec) {
    size_t num_circuits = circuits.size();

    // Generate instance polynomial and write it to transcript.
    std::vector<std::vector<Poly>> instance_polys_vec =
        GenerateInstancePolys(prover, instance_columns_vec);

    // Append leading zeros to each column of |instance_columns_vec|.
    size_t n = prover->pcs().N();
    for (size_t i = 0; i < num_circuits; ++i) {
      for (Evals& instance_column : instance_columns_vec[i]) {
        instance_column.evaluations().resize(n);
      }
    }

    // Generate advice poly by synthesizing circuit and write it to transcript.
    Synthesizer<PCS> synthesizer(num_circuits, &constraint_system);
    synthesizer.GenerateAdviceColumns(prover, circuits, instance_columns_vec);

    return Argument(num_circuits, fixed_columns, fixed_polys,
                    std::move(synthesizer).TakeAdviceColumnsVec(),
                    std::move(synthesizer).TakeAdviceBlindsVec(),
                    std::move(synthesizer).TakeChallenges(),
                    std::move(instance_columns_vec),
                    std::move(instance_polys_vec));
  }

  bool advice_transformed() const { return advice_transformed_; }

  // Generate a vector of advice coefficient-formed polynomials with a vector
  // of advice evaluation-formed columns. (a.k.a. Batch IFFT)
  // And for memory optimization, every evaluations of advice will be released
  // as soon as transforming it to coefficient form.
  void TransformAdvice(const Domain* domain) {
    CHECK(!advice_transformed_);
    advice_polys_vec_ = base::Map(
        advice_columns_vec_, [domain](std::vector<Evals>& advice_columns) {
          return base::Map(advice_columns, [domain](Evals& advice_column) {
            Poly poly = domain->IFFT(advice_column);
            // Release advice evals for memory optimization.
            advice_column = Evals::Zero();
            return poly;
          });
        });
    // Deallocate evaluations for memory optimization.
    advice_columns_vec_.clear();
    advice_transformed_ = true;
  }

  // Return tables including every type of polynomials in evaluation form.
  std::vector<RefTable<Evals>> ExportColumnTables() const {
    CHECK(!advice_transformed_);
    absl::Span<const Evals> fixed_columns =
        absl::MakeConstSpan(*fixed_columns_);

    return base::CreateVector(num_circuits_, [fixed_columns, this](size_t i) {
      absl::Span<const Evals> advice_columns =
          absl::MakeConstSpan(advice_columns_vec_[i]);
      absl::Span<const Evals> instance_columns =
          absl::MakeConstSpan(instance_columns_vec_[i]);
      return RefTable<Evals>(fixed_columns, advice_columns, instance_columns);
    });
  }

  // Return a table including every type of polynomials in coefficient form.
  std::vector<RefTable<Poly>> ExportPolyTables() const {
    CHECK(advice_transformed_);
    absl::Span<const Poly> fixed_polys = absl::MakeConstSpan(*fixed_polys_);
    return base::CreateVector(num_circuits_, [fixed_polys, this](size_t i) {
      absl::Span<const Poly> advice_polys =
          absl::MakeConstSpan(advice_polys_vec_[i]);
      absl::Span<const Poly> instance_polys =
          absl::MakeConstSpan(instance_polys_vec_[i]);
      return RefTable<Poly>(fixed_polys, advice_polys, instance_polys);
    });
  }

  const std::vector<F>& GetAdviceBlinds(size_t circuit_idx) const {
    CHECK_LT(circuit_idx, num_circuits_);
    return advice_blinds_vec_[circuit_idx];
  }

  const std::vector<F>& challenges() const { return challenges_; }

  std::vector<std::vector<LookupPermuted<Poly, Evals>>> CompressLookupStep(
      ProverBase<PCS>* prover, const ConstraintSystem<F>& constraint_system,
      const F& theta) const {
    std::vector<RefTable<Evals>> tables = ExportColumnTables();
    return BatchPermuteLookups(prover, constraint_system.lookups(), tables,
                               challenges_, theta);
  }

  StepReturns<PermutationCommitted<Poly>, LookupCommitted<Poly>,
              VanishingCommitted<PCS>>
  CommitCircuitStep(
      ProverBase<PCS>* prover, const ConstraintSystem<F>& constraint_system,
      const PermutationProvingKey<Poly, Evals>& permutation_proving_key,
      std::vector<std::vector<LookupPermuted<Poly, Evals>>>&&
          permuted_lookups_vec,
      const F& beta, const F& gamma) {
    std::vector<RefTable<Evals>> tables = ExportColumnTables();

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
        prover, proving_key, beta, gamma, theta, y, zeta, challenges_,
        committed.permutations(), committed.lookups_vec(), ExportPolyTables());
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
    std::vector<RefTable<Poly>> tables = ExportPolyTables();
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
    std::vector<RefTable<Poly>> tables = ExportPolyTables();
    const ConstraintSystem<F>& cs =
        proving_key.verifying_key().constraint_system();
    opening_points_set_.Insert(x);

    std::vector<crypto::PolynomialOpening<Poly>> ret;
    ret.reserve(GetNumOpenings(proving_key, evaluated, num_circuits_));
    for (size_t i = 0; i < num_circuits_; ++i) {
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
  // Generate a vector of instance coefficient-formed polynomials with a vector
  // of instance evaluation-formed columns. (a.k.a. Batch IFFT)
  static std::vector<std::vector<Poly>> GenerateInstancePolys(
      ProverBase<PCS>* prover,
      const std::vector<std::vector<Evals>>& instance_columns_vec) {
    size_t num_circuit = instance_columns_vec.size();
    CHECK_GT(num_circuit, size_t{0});
    size_t num_instance_columns = instance_columns_vec[0].size();
    if constexpr (PCS::kSupportsBatchMode && PCS::kQueryInstance) {
      size_t num_commitment = num_circuit * num_instance_columns;
      prover->pcs().SetBatchMode(num_commitment);
    }

    std::vector<std::vector<Poly>> instance_polys_vec;
    instance_polys_vec.reserve(num_circuit);
    for (size_t i = 0; i < num_circuit; ++i) {
      const std::vector<Evals>& instance_columns = instance_columns_vec[i];
      std::vector<Poly> instance_polys;
      instance_polys.reserve(num_instance_columns);
      for (size_t j = 0; j < num_instance_columns; ++j) {
        const Evals& instance_column = instance_columns[j];
        if constexpr (PCS::kQueryInstance && PCS::kSupportsBatchMode) {
          prover->BatchCommitAt(instance_column, i * num_instance_columns + j);
        } else if constexpr (PCS::kQueryInstance && !PCS::kSupportsBatchMode) {
          prover->CommitAndWriteToTranscript(instance_column);
        } else {
          for (const F& instance : instance_column.evaluations()) {
            CHECK(prover->GetWriter()->WriteToTranscript(instance));
          }
        }
        instance_polys.push_back(prover->domain()->IFFT(instance_column));
      }
      instance_polys_vec.push_back(std::move(instance_polys));
    }
    if constexpr (PCS::kSupportsBatchMode && PCS::kQueryInstance) {
      prover->RetrieveAndWriteBatchCommitmentsToTranscript();
    }
    return instance_polys_vec;
  }

  size_t num_circuits_ = 0;
  // not owned
  const std::vector<Evals>* fixed_columns_ = nullptr;
  // not owned
  const std::vector<Poly>* fixed_polys_ = nullptr;

  // NOTE(dongchangYoo): to optimize memory usage, release every advice
  // evaluations after generating an advice polynomial. That is, when
  // |advice_transformed_| is set to true, |advice_columns_vec_| is
  // released, and only |advice_polys_vec_| becomes available for use.
  bool advice_transformed_ = false;
  std::vector<std::vector<Evals>> advice_columns_vec_;
  std::vector<std::vector<Poly>> advice_polys_vec_;
  std::vector<std::vector<F>> advice_blinds_vec_;
  std::vector<F> challenges_;

  std::vector<std::vector<Evals>> instance_columns_vec_;
  std::vector<std::vector<Poly>> instance_polys_vec_;

  // NOTE(dongchangYoo): set of points which will be included to any openings.
  PointSet<F> opening_points_set_;
};

}  // namespace tachyon::zk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_ARGUMENT_H_
