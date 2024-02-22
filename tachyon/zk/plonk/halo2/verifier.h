// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_HALO2_VERIFIER_H_
#define TACHYON_ZK_PLONK_HALO2_VERIFIER_H_

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest_prod.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/crypto/commitments/polynomial_openings.h"
#include "tachyon/zk/base/entities/verifier_base.h"
#include "tachyon/zk/lookup/lookup_verification.h"
#include "tachyon/zk/plonk/halo2/proof_reader.h"
#include "tachyon/zk/plonk/keys/verifying_key.h"
#include "tachyon/zk/plonk/permutation/permutation_verification.h"
#include "tachyon/zk/plonk/vanishing/vanishing_verification_evaluator.h"

namespace tachyon::zk::plonk::halo2 {

template <typename PCS>
class Verifier : public VerifierBase<PCS> {
 public:
  using F = typename PCS::Field;
  using Commitment = typename PCS::Commitment;
  using Evals = typename PCS::Evals;
  using Poly = typename PCS::Poly;
  using Coefficients = typename Poly::Coefficients;
  using Opening = crypto::PolynomialOpening<Poly, Commitment>;

  using VerifierBase<PCS>::VerifierBase;

  [[nodiscard]] bool VerifyProof(
      const VerifyingKey<F, Commitment>& vkey,
      const std::vector<std::vector<Evals>>& instance_columns_vec) {
    return VerifyProofForTesting(vkey, instance_columns_vec, nullptr, nullptr);
  }

 private:
  FRIEND_TEST(SimpleCircuitTest, Verify);
  FRIEND_TEST(SimpleV1CircuitTest, Verify);
  FRIEND_TEST(SimpleLookupCircuitTest, Verify);
  FRIEND_TEST(SimpleLookupV1CircuitTest, Verify);
  template <typename>
  FRIEND_TEST(ShuffleCircuitTest, Verify);

  bool VerifyProofForTesting(
      const VerifyingKey<F, Commitment>& vkey,
      const std::vector<std::vector<Evals>>& instance_columns_vec,
      Proof<F, Commitment>* proof_out, F* expected_h_eval_out) {
    if (!ValidateInstanceColumnsVec(vkey, instance_columns_vec)) return false;

    std::vector<std::vector<Commitment>> instance_commitments_vec;
    if constexpr (PCS::kQueryInstance) {
      instance_commitments_vec = CommitColumnsVec(instance_columns_vec);
    } else {
      instance_commitments_vec.resize(instance_columns_vec.size());
    }

    crypto::TranscriptReader<Commitment>* transcript = this->GetReader();
    CHECK(transcript->WriteToTranscript(vkey.transcript_repr()));

    if constexpr (PCS::kQueryInstance) {
      WriteCommitmentsVecToTranscript(transcript, instance_commitments_vec);
    } else {
      WriteColumnsVecToTranscript(transcript, instance_columns_vec);
    }

    ProofReader<PCS> proof_reader(vkey, transcript,
                                  instance_commitments_vec.size());
    Proof<F, Commitment>& proof = proof_reader.proof();
    proof_reader.ReadAdviceCommitmentsVecAndChallenges();
    proof_reader.ReadTheta();
    proof_reader.ReadLookupPermutedCommitments();
    proof_reader.ReadBetaAndGamma();
    proof_reader.ReadPermutationProductCommitments();
    proof_reader.ReadLookupProductCommitments();
    proof_reader.ReadVanishingRandomPolyCommitment();
    proof_reader.ReadY();
    proof_reader.ReadVanishingHPolyCommitments();
    proof_reader.ReadX();
    if constexpr (PCS::kQueryInstance) {
      proof_reader.ReadInstanceEvalsIfQueryInstance();
    } else {
      proof_reader.ReadInstanceEvalsIfNoQueryInstance();
      proof.instance_evals_vec =
          ComputeInstanceEvalsVec(vkey, instance_columns_vec, proof.x);
    }
    proof_reader.ReadAdviceEvals();
    proof_reader.ReadFixedEvals();
    proof_reader.ReadVanishingRandomEval();
    proof_reader.ReadCommonPermutationEvals();
    proof_reader.ReadPermutationEvals();
    proof_reader.ReadLookupEvals();
    CHECK(proof_reader.Done());

    if (proof_out) {
      *proof_out = proof;
    }

    ComputeAuxValues(vkey.constraint_system(), proof);

    return DoVerify(instance_commitments_vec, vkey, proof, expected_h_eval_out);
  }

  void ComputeAuxValues(const ConstraintSystem<F>& constraint_system,
                        Proof<F, Commitment>& proof) const {
    RowIndex blinding_factors = constraint_system.ComputeBlindingFactors();
    std::vector<F> l_evals = this->domain_->EvaluatePartialLagrangeCoefficients(
        proof.x, base::Range<int32_t, /*IsStartInclusive=*/true,
                             /*IsEndInclusive=*/true>(
                     -static_cast<int32_t>(blinding_factors + 1), 0));
    proof.l_first = l_evals[1 + blinding_factors];
    proof.l_blind = std::accumulate(
        l_evals.begin() + 1, l_evals.begin() + 1 + blinding_factors, F::Zero(),
        [](F& acc, const F& eval) { return acc += eval; });
    proof.l_last = l_evals[0];

    proof.x_next = Rotation::Next().RotateOmega(this->domain(), proof.x);
    proof.x_prev = Rotation::Prev().RotateOmega(this->domain(), proof.x);
    proof.x_last =
        Rotation(-(blinding_factors + 1)).RotateOmega(this->domain(), proof.x);
    proof.x_n = proof.x.Pow(this->pcs_.N());
  }

  bool ValidateInstanceColumnsVec(
      const VerifyingKey<F, Commitment>& vkey,
      const std::vector<std::vector<Evals>>& instance_columns_vec) const {
    size_t num_instance_columns =
        vkey.constraint_system().num_instance_columns();
    auto check_num_instance_columns =
        [num_instance_columns](const std::vector<Evals>& instance_columns) {
          if (instance_columns.size() != num_instance_columns) {
            LOG(ERROR) << "The size of instance columns doesn't match with "
                          "constraint system";
            return false;
          }
          return true;
        };

    // NOTE(chokobole): It's safe to downcast because domain is already checked.
    RowIndex max_rows = static_cast<RowIndex>(this->pcs_.N()) -
                        (vkey.constraint_system().ComputeBlindingFactors() + 1);
    auto check_rows = [max_rows](const Evals& instance_columns) {
      if (instance_columns.NumElements() > size_t{max_rows}) {
        LOG(ERROR) << "Too many number of elements in instance column";
        return false;
      }
      return true;
    };

    return std::all_of(instance_columns_vec.begin(), instance_columns_vec.end(),
                       [&check_num_instance_columns, &check_rows](
                           const std::vector<Evals>& instance_columns) {
                         if (!check_num_instance_columns(instance_columns))
                           return false;
                         return std::all_of(instance_columns.begin(),
                                            instance_columns.end(), check_rows);
                       });
  }

  std::vector<Commitment> CommitColumns(const std::vector<Evals>& columns) {
    return base::Map(columns, [this](const Evals& column) {
      std::vector<F> expanded_evals = column.evaluations();
      expanded_evals.resize(this->pcs_.N());
      Commitment c;
      CHECK(this->pcs_.CommitLagrange(Evals(std::move(expanded_evals)), &c));
      return c;
    });
  }

  std::vector<std::vector<Commitment>> CommitColumnsVec(
      const std::vector<std::vector<Evals>>& columns_vec) {
    return base::Map(columns_vec, [this](const std::vector<Evals>& columns) {
      return CommitColumns(columns);
    });
  }

  static void WriteCommitmentsVecToTranscript(
      crypto::TranscriptReader<Commitment>* transcript,
      const std::vector<std::vector<Commitment>>& commitments_vec) {
    for (const std::vector<Commitment>& commitments : commitments_vec) {
      for (const Commitment& commitment : commitments) {
        CHECK(transcript->WriteToTranscript(commitment));
      }
    }
  }

  static void WriteColumnsVecToTranscript(
      crypto::TranscriptReader<Commitment>* transcript,
      const std::vector<std::vector<Evals>>& columns_vec) {
    for (const std::vector<Evals>& columns : columns_vec) {
      for (const Evals& column : columns) {
        for (const F& value : column.evaluations()) {
          CHECK(transcript->WriteToTranscript(value));
        }
      }
    }
  }

  static RowIndex ComputeMaxRow(const std::vector<Evals>& columns) {
    if (columns.empty()) return 0;
    std::vector<RowIndex> rows = base::Map(columns, [](const Evals& evals) {
      // NOTE(chokobole): It's safe to downcast because domain is already
      // checked.
      return static_cast<RowIndex>(evals.NumElements());
    });
    return *std::max_element(rows.begin(), rows.end());
  }

  static F ComputeInstanceEval(const std::vector<Evals>& instance_columns,
                               const InstanceQueryData& instance_query,
                               const std::vector<F>& partial_lagrange_coeffs,
                               size_t max_rotation) {
    const std::vector<F>& instances =
        instance_columns[instance_query.column().index()].evaluations();
    size_t offset = max_rotation - instance_query.rotation().value();
    absl::Span<const F> sub_partial_lagrange_coeffs(partial_lagrange_coeffs);
    sub_partial_lagrange_coeffs =
        sub_partial_lagrange_coeffs.subspan(offset, instances.size());
    return F::SumOfProductsSerial(instances, sub_partial_lagrange_coeffs);
  }

  static std::vector<F> ComputeInstanceEvals(
      const std::vector<Evals>& instance_columns,
      const std::vector<InstanceQueryData>& instance_queries,
      const std::vector<F>& partial_lagrange_coeffs, size_t max_rotation) {
    return base::Map(instance_queries,
                     [&instance_columns, &partial_lagrange_coeffs,
                      max_rotation](const InstanceQueryData& instance_query) {
                       return ComputeInstanceEval(
                           instance_columns, instance_query,
                           partial_lagrange_coeffs, max_rotation);
                     });
  }

  std::vector<std::vector<F>> ComputeInstanceEvalsVec(
      const VerifyingKey<F, Commitment>& vkey,
      const std::vector<std::vector<Evals>>& instance_columns_vec, const F& x) {
    struct RotationRange {
      int32_t min = 0;
      int32_t max = 0;
    };

    const std::vector<InstanceQueryData>& instance_queries =
        vkey.constraint_system().instance_queries();
    RotationRange range = std::accumulate(
        instance_queries.begin(), instance_queries.end(), RotationRange(),
        [](RotationRange& range, const InstanceQueryData& instance) {
          int32_t rotation_value = instance.rotation().value();
          if (rotation_value < range.min) {
            range.min = rotation_value;
          } else if (rotation_value > range.max) {
            range.max = rotation_value;
          }
          return range;
        });

    std::vector<RowIndex> max_instances_rows =
        base::Map(instance_columns_vec, &ComputeMaxRow);
    auto max_instances_row_it =
        std::max_element(max_instances_rows.begin(), max_instances_rows.end());
    RowIndex max_instances_row =
        max_instances_row_it != max_instances_rows.end() ? *max_instances_row_it
                                                         : 0;
    std::vector<F> partial_lagrange_coeffs =
        this->domain_->EvaluatePartialLagrangeCoefficients(
            x, base::Range<int32_t>(-range.max,
                                    static_cast<int32_t>(max_instances_row) +
                                        std::abs(range.min)));

    return base::Map(instance_columns_vec,
                     [&instance_queries, &partial_lagrange_coeffs,
                      &range](const std::vector<Evals>& instance_columns) {
                       return ComputeInstanceEvals(
                           instance_columns, instance_queries,
                           partial_lagrange_coeffs, range.max);
                     });
  }

  F ComputeExpectedHEval(size_t num_circuits,
                         const VerifyingKey<F, Commitment>& vkey,
                         const Proof<F, Commitment>& proof) {
    const ConstraintSystem<F>& constraint_system = vkey.constraint_system();
    std::vector<F> expressions;
    const std::vector<Gate<F>>& gates = constraint_system.gates();
    const std::vector<LookupArgument<F>>& lookups = constraint_system.lookups();
    size_t polys_size = std::accumulate(gates.begin(), gates.end(), 0,
                                        [](size_t acc, const Gate<F>& gate) {
                                          return acc + gate.polys().size();
                                        });
    size_t expressions_size =
        num_circuits *
        (polys_size +
         GetSizeOfPermutationVerificationExpressions(constraint_system) +
         lookups.size() * GetSizeOfLookupVerificationExpressions());
    expressions.reserve(expressions_size);
    for (size_t i = 0; i < num_circuits; ++i) {
      VanishingVerificationData<F> data = proof.ToVanishingVerificationData(i);
      VanishingVerificationEvaluator<F> vanishing_verification_evaluator(data);
      for (const Gate<F>& gate : gates) {
        for (const std::unique_ptr<Expression<F>>& poly : gate.polys()) {
          expressions.push_back(
              poly->Evaluate(&vanishing_verification_evaluator));
        }
      }

      std::vector<F> permutation_expressions =
          CreatePermutationVerificationExpressions(
              proof.ToPermutationVerificationData(i), constraint_system);
      expressions.insert(
          expressions.end(),
          std::make_move_iterator(permutation_expressions.begin()),
          std::make_move_iterator(permutation_expressions.end()));

      for (size_t j = 0; j < lookups.size(); ++j) {
        const LookupArgument<F>& lookup = lookups[j];
        std::vector<F> lookup_expressions = CreateLookupVerificationExpressions(
            proof.ToLookupVerificationData(i, j), lookup);
        expressions.insert(expressions.end(),
                           std::make_move_iterator(lookup_expressions.begin()),
                           std::make_move_iterator(lookup_expressions.end()));
      }
    }
    DCHECK_EQ(expressions.size(), expressions_size);
    F expected_h_eval =
        F::template LinearCombination</*forward=*/true>(expressions, proof.y);
    return expected_h_eval /= (proof.x_n - F::One());
  }

  size_t GetSizeOfAdviceInstanceColumnQueries(
      const ConstraintSystem<F>& constraint_system) {
    const std::vector<AdviceQueryData>& advice_queries =
        constraint_system.advice_queries();
    size_t size = advice_queries.size();
    if constexpr (PCS::kQueryInstance) {
      const std::vector<InstanceQueryData>& instance_queries =
          constraint_system.instance_queries();
      size += instance_queries.size();
    }
    return size;
  }

  template <ColumnType C>
  void CreateColumnQueries(const std::vector<QueryData<C>>& queries,
                           const std::vector<Commitment>& commitments,
                           const std::vector<F>& evals,
                           const Proof<F, Commitment>& proof,
                           std::vector<Opening>& openings,
                           std::vector<F>& points) const {
    for (size_t i = 0; i < queries.size(); ++i) {
      const QueryData<C>& query = queries[i];
      const ColumnKey<C>& column = query.column();
      points.push_back(query.rotation().RotateOmega(this->domain(), proof.x));
      openings.emplace_back(
          base::Ref<const Commitment>(&commitments[column.index()]),
          base::DeepRef<const F>(&points.back()), evals[i]);
    }
  }

  bool DoVerify(
      const std::vector<std::vector<Commitment>>& instance_commitments_vec,
      const VerifyingKey<F, Commitment>& vkey,
      const Proof<F, Commitment>& proof, F* expected_h_eval_out) {
    std::vector<Opening> queries;
    size_t num_circuits = instance_commitments_vec.size();

    const ConstraintSystem<F>& constraint_system = vkey.constraint_system();
    const std::vector<LookupArgument<F>>& lookups = constraint_system.lookups();
    const std::vector<FixedQueryData>& fixed_queries =
        constraint_system.fixed_queries();
    const std::vector<Commitment>& common_permutation_commitments =
        vkey.permutation_verifying_key().commitments();
    size_t queries_size =
        num_circuits *
            (GetSizeOfAdviceInstanceColumnQueries(constraint_system) +
             GetSizeOfPermutationVerifierQueries(constraint_system) +
             lookups.size() * GetSizeOfLookupVerifierQueries()) +
        fixed_queries.size() + common_permutation_commitments.size() + 2;
    queries.reserve(queries_size);

    std::vector<F> points;
    size_t points_size =
        num_circuits * GetSizeOfAdviceInstanceColumnQueries(constraint_system) +
        fixed_queries.size();
    points.reserve(points_size);

    for (size_t i = 0; i < num_circuits; ++i) {
      if constexpr (PCS::kQueryInstance) {
        const std::vector<InstanceQueryData>& instance_queries =
            constraint_system.instance_queries();
        CreateColumnQueries(instance_queries, instance_commitments_vec[i],
                            proof.instance_evals_vec[i], proof, queries,
                            points);
      }
      const std::vector<AdviceQueryData>& advice_queries =
          constraint_system.advice_queries();
      CreateColumnQueries(advice_queries, proof.advices_commitments_vec[i],
                          proof.advice_evals_vec[i], proof, queries, points);

      std::vector<Opening> permutation_queries = CreatePermutationQueries<PCS>(
          proof.ToPermutationVerificationData(i), constraint_system);
      queries.insert(queries.end(),
                     std::make_move_iterator(permutation_queries.begin()),
                     std::make_move_iterator(permutation_queries.end()));

      for (size_t j = 0; j < lookups.size(); ++j) {
        std::vector<Opening> lookup_queries =
            CreateLookupQueries<PCS>(proof.ToLookupVerificationData(i, j));
        queries.insert(queries.end(),
                       std::make_move_iterator(lookup_queries.begin()),
                       std::make_move_iterator(lookup_queries.end()));
      }
    }

    CreateColumnQueries(fixed_queries, vkey.fixed_commitments(),
                        proof.fixed_evals, proof, queries, points);

    for (size_t i = 0; i < proof.common_permutation_evals.size(); ++i) {
      queries.emplace_back(
          base::Ref<const Commitment>(&common_permutation_commitments[i]),
          base::DeepRef<const F>(&proof.x), proof.common_permutation_evals[i]);
    }

    // TODO(chokobole): Remove |ToAffine()| since this assumes commitment is an
    // elliptic curve point.
    Commitment h_commitment =
        Commitment::template LinearCombination</*forward=*/false>(
            proof.vanishing_h_poly_commitments, proof.x_n)
            .ToAffine();

    F expected_h_eval = ComputeExpectedHEval(num_circuits, vkey, proof);

    if (expected_h_eval_out) {
      *expected_h_eval_out = expected_h_eval;
    }

    queries.emplace_back(base::Ref<const Commitment>(&h_commitment),
                         base::DeepRef<const F>(&proof.x), expected_h_eval);
    queries.emplace_back(
        base::Ref<const Commitment>(&proof.vanishing_random_poly_commitment),
        base::DeepRef<const F>(&proof.x), proof.vanishing_random_eval);
    DCHECK_EQ(queries.size(), queries_size);
    DCHECK_EQ(points.size(), points_size);
    return this->pcs_.VerifyOpeningProof(queries, this->GetReader());
  }
};

}  // namespace tachyon::zk::plonk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_VERIFIER_H_
