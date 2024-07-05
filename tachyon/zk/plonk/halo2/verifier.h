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
#include "tachyon/zk/lookup/halo2/opening_point_set.h"
#include "tachyon/zk/lookup/halo2/utils.h"
#include "tachyon/zk/plonk/halo2/proof_reader.h"
#include "tachyon/zk/plonk/keys/verifying_key.h"
#include "tachyon/zk/plonk/permutation/permutation_utils.h"
#include "tachyon/zk/plonk/permutation/permutation_verifier.h"
#include "tachyon/zk/plonk/vanishing/vanishing_utils.h"
#include "tachyon/zk/plonk/vanishing/vanishing_verifier.h"

namespace tachyon::zk::plonk {

template <typename TestArguments, typename TestData>
class CircuitTest;

namespace halo2 {

template <typename PCS, typename _LS>
class Verifier : public VerifierBase<PCS> {
 public:
  using F = typename PCS::Field;
  using Commitment = typename PCS::Commitment;
  using Evals = typename PCS::Evals;
  using Poly = typename PCS::Poly;
  using Coefficients = typename Poly::Coefficients;
  using Opening = crypto::PolynomialOpening<Poly, Commitment>;
  using LS = _LS;
  using LookupVerifier = typename LS::Verifier;
  using Proof = typename LS::Proof;

  using VerifierBase<PCS>::VerifierBase;

  [[nodiscard]] bool VerifyProof(
      const VerifyingKey<F, Commitment>& vkey,
      const std::vector<std::vector<Evals>>& instance_columns_vec) {
    return VerifyProofForTesting(vkey, instance_columns_vec, nullptr, nullptr);
  }

 private:
  template <typename TestArguments, typename TestData>
  friend class plonk::CircuitTest;

  bool VerifyProofForTesting(
      const VerifyingKey<F, Commitment>& vkey,
      const std::vector<std::vector<Evals>>& instance_columns_vec,
      Proof* proof_out, F* expected_h_eval_out) {
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

    ProofReader<PCS, LS> proof_reader(vkey, transcript,
                                      instance_commitments_vec.size());
    Proof& proof = proof_reader.proof();
    proof_reader.ReadAdviceCommitmentsVecAndChallenges();
    proof_reader.ReadTheta();
    proof_reader.ReadLookupPreparedCommitments();
    proof_reader.ReadBetaAndGamma();
    proof_reader.ReadPermutationProductCommitments();
    proof_reader.ReadLookupGrandCommitments();
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
                        Proof& proof) const {
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

    RowIndex usable_rows =
        this->GetUsableRows(vkey.constraint_system().ComputeBlindingFactors());
    auto check_rows = [usable_rows](const Evals& instance_columns) {
      if (instance_columns.NumElements() > size_t{usable_rows}) {
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
      return this->Commit(Evals(expanded_evals));
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

  F EvaluateH(
      const std::vector<std::vector<Commitment>>& instance_commitments_vec,
      const VerifyingKey<F, Commitment>& vkey, const Proof& proof) {
    size_t num_circuits = proof.advices_commitments_vec.size();
    const ConstraintSystem<F>& constraint_system = vkey.constraint_system();
    size_t size =
        GetNumVanishingEvals(num_circuits, constraint_system.gates()) +
        GetNumPermutationEvals(
            num_circuits, proof.permutation_product_commitments_vec[0].size()) +
        lookup::halo2::GetNumEvals(LS::type, num_circuits,
                                   constraint_system.lookups().size());
    std::vector<F> evals;
    evals.reserve(size);

    LValues<F> l_values(proof.l_first, proof.l_blind, proof.l_last);
    for (size_t i = 0; i < num_circuits; ++i) {
      VanishingVerifierData<F, Commitment> vanishing_verifier_data =
          proof.ToVanishingVerifierData(i, vkey.fixed_commitments(),
                                        instance_commitments_vec[i]);
      VanishingVerifier<F, Commitment> vanishing_verifier(
          vanishing_verifier_data);
      vanishing_verifier.Evaluate(constraint_system, evals);

      PermutationVerifierData<F, Commitment> permutation_verifier_data =
          proof.ToPermutationVerifierData(
              i, vkey.permutation_verifying_key().commitments());
      PermutationVerifier<F, Commitment> permutation_verifier(
          permutation_verifier_data);
      permutation_verifier.Evaluate(constraint_system, proof.x, l_values,
                                    evals);

      LookupVerifier lookup_verifier(proof, i, l_values);
      lookup_verifier.Evaluate(constraint_system.lookups(), evals);
    }
    DCHECK_EQ(evals.size(), size);
    F expected_h_eval =
        F::template LinearCombination</*forward=*/true>(evals, proof.y);
    CHECK(expected_h_eval /= (proof.x_n - F::One()));
    return expected_h_eval;
  }

  std::vector<Opening> Open(
      const std::vector<std::vector<Commitment>>& instance_commitments_vec,
      const VerifyingKey<F, Commitment>& vkey, const Proof& proof,
      Commitment& expected_h_commitment, const F& expected_h_eval) {
    size_t num_circuits = proof.advices_commitments_vec.size();
    const ConstraintSystem<F>& constraint_system = vkey.constraint_system();
    size_t size =
        GetNumVanishingOpenings<PCS>(
            num_circuits, constraint_system.advice_queries().size(),
            constraint_system.instance_queries().size(),
            constraint_system.fixed_queries().size()) +
        GetNumPermutationOpenings(
            num_circuits, proof.permutation_product_commitments_vec[0].size(),
            vkey.permutation_verifying_key().commitments().size()) +
        lookup::halo2::GetNumOpenings(LS::type, num_circuits,
                                      constraint_system.lookups().size());
    std::vector<Opening> openings;
    openings.reserve(size);

    PermutationOpeningPointSet<F> permutation_point_set(proof.x, proof.x_next,
                                                        proof.x_last);
    lookup::halo2::OpeningPointSet<F> lookup_point_set(proof.x, proof.x_prev,
                                                       proof.x_next);

    for (size_t i = 0; i < num_circuits; ++i) {
      VanishingVerifierData<F, Commitment> vanishing_verifier_data =
          proof.ToVanishingVerifierData(i, vkey.fixed_commitments(),
                                        instance_commitments_vec[i]);
      VanishingVerifier<F, Commitment> vanishing_verifier(
          vanishing_verifier_data);
      vanishing_verifier.template OpenAdviceInstanceColumns<PCS, Poly>(
          this->domain(), proof.x, constraint_system, openings);

      PermutationVerifierData<F, Commitment> permutation_verifier_data =
          proof.ToPermutationVerifierData(
              i, vkey.permutation_verifying_key().commitments());
      PermutationVerifier<F, Commitment> permutation_verifier(
          permutation_verifier_data);
      permutation_verifier.template Open<Poly>(permutation_point_set, openings);

      LookupVerifier lookup_verifier(proof, i);
      lookup_verifier.Open(lookup_point_set, openings);

      if (i == num_circuits - 1) {
        vanishing_verifier.template OpenFixedColumns<Poly>(
            this->domain(), proof.x, constraint_system, openings);

        permutation_verifier.template OpenPermutationProvingKey<Poly>(proof.x,
                                                                      openings);

        vanishing_verifier.template Open<Poly>(proof.x, proof.x_n,
                                               expected_h_commitment,
                                               expected_h_eval, openings);
      }
    }

    DCHECK_EQ(openings.size(), size);
    return openings;
  }

  bool DoVerify(
      const std::vector<std::vector<Commitment>>& instance_commitments_vec,
      const VerifyingKey<F, Commitment>& vkey, const Proof& proof,
      F* expected_h_eval_out) {
    F expected_h_eval = EvaluateH(instance_commitments_vec, vkey, proof);
    if (expected_h_eval_out) {
      *expected_h_eval_out = expected_h_eval;
    }

    // TODO(chokobole): Remove |ToAffine()| since this assumes commitment is an
    // elliptic curve point.
    Commitment expected_h_commitment;
    std::vector<Opening> openings =
        Open(instance_commitments_vec, vkey, proof, expected_h_commitment,
             expected_h_eval);

    return this->pcs_.VerifyOpeningProof(openings, this->GetReader());
  }
};

}  // namespace halo2
}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_HALO2_VERIFIER_H_
