// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_HALO2_VERIFIER_H_
#define TACHYON_ZK_PLONK_HALO2_VERIFIER_H_

#include <functional>
#include <utility>
#include <vector>

#include "absl/functional/bind_front.h"
#include "gtest/gtest_prod.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/zk/base/entities/verifier_base.h"
#include "tachyon/zk/plonk/halo2/proof_reader.h"
#include "tachyon/zk/plonk/keys/verifying_key.h"

namespace tachyon::zk::halo2 {

template <typename PCSTy>
class Verifier : public VerifierBase<PCSTy> {
 public:
  using F = typename PCSTy::Field;
  using Commitment = typename PCSTy::Commitment;
  using Evals = typename PCSTy::Evals;
  using Poly = typename PCSTy::Poly;
  using Coefficients = typename Poly::Coefficients;

  using VerifierBase<PCSTy>::VerifierBase;

  [[nodiscard]] bool VerifyProof(
      const VerifyingKey<PCSTy>& vkey,
      const std::vector<std::vector<Evals>>& instance_columns_vec) {
    return VerifyProofForTesting(vkey, instance_columns_vec, nullptr);
  }

 private:
  FRIEND_TEST(SimpleCircuitTest, Verify);
  FRIEND_TEST(SimpleLookupCircuitTest, Verify);

  bool VerifyProofForTesting(
      const VerifyingKey<PCSTy>& vkey,
      const std::vector<std::vector<Evals>>& instance_columns_vec,
      Proof<F, Commitment>* proof_out) {
    if (!ValidateInstanceColumnsVec(vkey, instance_columns_vec)) return false;

    std::vector<std::vector<Commitment>> instance_commitments_vec;
    if constexpr (PCSTy::kQueryInstance) {
      instance_commitments_vec = CommitColumnsVec(vkey, instance_columns_vec);
    } else {
      instance_commitments_vec.resize(instance_columns_vec.size());
    }

    crypto::TranscriptReader<Commitment>* transcript = this->GetReader();
    CHECK(transcript->WriteToTranscript(vkey.transcript_repr()));

    if constexpr (PCSTy::kQueryInstance) {
      WriteCommitmentsVecToTranscript(transcript, instance_commitments_vec);
    } else {
      WriteColumnsVecToTranscript(transcript, instance_columns_vec);
    }

    ProofReader<PCSTy> proof_reader(vkey, transcript,
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
    if constexpr (PCSTy::kQueryInstance) {
      proof_reader.ReadInstanceEvalsIfQueryInstance();
    } else {
      proof_reader.ReadInstanceEvalsIfNoQueryInstance();
      proof.instance_evals_vec =
          ComputeInstanceEvalsVec(vkey, instance_columns_vec, proof.x);
    }
    proof_reader.ReadAdviceEvals();
    proof_reader.ReadFixedEvals();
    proof_reader.ReadVanishingEval();
    proof_reader.ReadCommonPermutationEvals();
    proof_reader.ReadPermutationEvals();
    proof_reader.ReadLookupEvals();

    if (proof_out) {
      *proof_out = proof;
    }
    return true;
  }

  bool ValidateInstanceColumnsVec(
      const VerifyingKey<PCSTy>& vkey,
      const std::vector<std::vector<Evals>>& instance_columns_vec) {
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

    size_t max_rows = this->pcs_.N() -
                      (vkey.constraint_system().ComputeBlindingFactors() + 1);
    auto check_rows = [max_rows](const Evals& instance_columns) {
      if (instance_columns.NumElements() > max_rows) {
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
      CHECK(this->pcs_.CommitLagrange(Evals(std::move(expanded_evals), &c)));
      return c;
    });
  }

  std::vector<std::vector<Commitment>> CommitColumnsVec(
      const std::vector<std::vector<Evals>>& columns_vec) {
    return base::Map(columns_vec, absl::bind_front(&CommitColumns, this));
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

  static size_t ComputeMaxRow(const std::vector<Evals>& columns) {
    if (columns.empty()) return 0;
    std::vector<size_t> rows = base::Map(
        columns, [](const Evals& evals) { return evals.NumElements(); });
    return *std::max_element(rows.begin(), rows.end());
  }

  static F ComputeInstanceEval(const std::vector<Evals>& instance_columns,
                               const InstanceQueryData& instance_query,
                               const std::vector<F>& partial_lagrange_coeffs,
                               size_t max_rotation) {
    const std::vector<F>& instances =
        instance_columns[instance_query.column().index()].evaluations();
    size_t offset = max_rotation - instance_query.rotation().value();
    absl::Span<const F> sub_partial_lagrange_coeffs =
        absl::MakeConstSpan(partial_lagrange_coeffs);
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
      const VerifyingKey<PCSTy>& vkey,
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

    std::vector<size_t> max_instances_rows =
        base::Map(instance_columns_vec, &ComputeMaxRow);
    auto max_instances_row_it =
        std::max_element(max_instances_rows.begin(), max_instances_rows.end());
    size_t max_instances_row = max_instances_row_it != max_instances_rows.end()
                                   ? *max_instances_row_it
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
};

}  // namespace tachyon::zk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_VERIFIER_H_
