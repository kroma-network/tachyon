// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_CRYPTO_COMMITMENTS_FRI_TWO_ADIC_FRI_H_
#define TACHYON_CRYPTO_COMMITMENTS_FRI_TWO_ADIC_FRI_H_

#include <algorithm>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"

#include "tachyon/base/bits.h"
#include "tachyon/base/parallelize.h"
#include "tachyon/base/profiler.h"
#include "tachyon/crypto/commitments/fri/fri_proof.h"
#include "tachyon/crypto/commitments/fri/prove.h"
#include "tachyon/crypto/commitments/fri/two_adic_multiplicative_coset.h"
#include "tachyon/crypto/commitments/fri/verify.h"
#include "tachyon/crypto/commitments/mixed_matrix_commitment_scheme.h"
#include "tachyon/math/finite_fields/extension_field_traits_forward.h"
#include "tachyon/math/geometry/dimensions.h"
#include "tachyon/math/matrix/matrix_types.h"
#include "tachyon/math/matrix/matrix_utils.h"
#include "tachyon/math/polynomials/univariate/evaluations_utils.h"

namespace tachyon {
namespace c::crypto {

template <typename ExtF, typename InputMMCS, typename ChallengeMMCS,
          typename Challenger>
class TwoAdicFRIImpl;

}  // namespace c::crypto

namespace crypto {

template <typename ExtF, typename _InputMMCS, typename _ChallengeMMCS,
          typename Challenger>
class TwoAdicFRI {
 public:
  using InputMMCS = _InputMMCS;
  using ChallengeMMCS = _ChallengeMMCS;

  using F = typename math::ExtensionFieldTraits<ExtF>::BaseField;
  using Domain = TwoAdicMultiplicativeCoset<F>;
  using Commitment = typename InputMMCS::Commitment;
  using ProverData = typename InputMMCS::ProverData;
  using Proof = typename InputMMCS::Proof;
  using InputProof = std::vector<BatchOpening<TwoAdicFRI>>;
  using FRIProof = crypto::FRIProof<TwoAdicFRI>;

  using OpeningPointsForRound = std::vector<std::vector<ExtF>>;
  using OpeningPoints = std::vector<OpeningPointsForRound>;

  using OpenedValuesForMat = std::vector<std::vector<ExtF>>;
  using OpenedValuesForRound = std::vector<OpenedValuesForMat>;
  using OpenedValues = std::vector<OpenedValuesForRound>;

  TwoAdicFRI() = default;
  TwoAdicFRI(InputMMCS&& mmcs, FRIConfig<ChallengeMMCS>&& fri)
      : mmcs_(std::move(mmcs)), fri_(std::move(fri)) {}

  Domain GetNaturalDomainForDegree(size_t size) {
    uint32_t log_n = base::bits::CheckedLog2(size);
    return Domain(log_n, F::One());
  }

  [[nodiscard]] bool Commit(const std::vector<Domain>& cosets,
                            std::vector<math::RowMajorMatrix<F>>& matrices,
                            Commitment* commitment, ProverData* prover_data) {
    TRACE_EVENT("ProofGeneration", "TwoAdicFRI::Commit");
    std::vector<math::RowMajorMatrix<F>> ldes =
        base::Map(cosets, [this, &matrices](size_t i, const Domain& coset) {
          math::RowMajorMatrix<F>& mat = matrices[i];
          CHECK_EQ(coset.domain()->size(), static_cast<size_t>(mat.rows()));
          math::RowMajorMatrix<F> ret = coset.domain()->CosetLDEBatch(
              mat, fri_.log_blowup,
              F::FromMontgomery(F::Config::kSubgroupGenerator) *
                  coset.domain()->offset_inv());
          ReverseMatrixIndexBits(ret);
          return ret;
        });
    return mmcs_.Commit(std::move(ldes), commitment, prover_data);
  }

  [[nodiscard]] bool CreateOpeningProof(
      const std::vector<ProverData>& prover_data_by_round,
      const OpeningPoints& points_by_round, Challenger& challenger,
      OpenedValues* opened_values_out, FRIProof* proof) {
    TRACE_EVENT("ProofGeneration", "TwoAdicFRI::CreateOpeningProof");
    ExtF alpha = challenger.template SampleExtElement<ExtF>();
    VLOG(2) << "FRI(alpha): " << alpha.ToHexString(true);
    size_t num_rounds = prover_data_by_round.size();

    size_t global_max_num_rows = 0;
    std::vector<absl::Span<const math::RowMajorMatrix<F>>> matrices_by_round =
        base::Map(prover_data_by_round,
                  [this, &global_max_num_rows](const ProverData& prover_data) {
                    global_max_num_rows = std::max(
                        global_max_num_rows, mmcs_.GetMaxRowSize(prover_data));
                    return absl::MakeConstSpan(mmcs_.GetMatrices(prover_data));
                  });
    uint32_t log_global_max_num_rows =
        base::bits::CheckedLog2(global_max_num_rows);

    // For each unique opening point z, we will find the largest degree bound
    // for that point, and precompute 1/(X - z) for the largest subgroup (in
    // bitrev order).
    absl::flat_hash_map<ExtF, std::vector<ExtF>> inv_denoms =
        ComputeInverseDenominators(
            matrices_by_round, points_by_round,
            F::FromMontgomery(F::Config::kSubgroupGenerator));

    std::array<std::vector<ExtF>, 32> reduced_openings;
    std::array<size_t, 32> num_reduced = {};

    OpenedValues opened_values(num_rounds);
    for (size_t round = 0; round < num_rounds; ++round) {
      absl::Span<const math::RowMajorMatrix<F>> matrices =
          matrices_by_round[round];
      const OpeningPointsForRound& points = points_by_round[round];
      OpenedValuesForRound opened_values_for_round(matrices.size());
      for (size_t matrix_idx = 0; matrix_idx < matrices.size(); ++matrix_idx) {
        const math::RowMajorMatrix<F>& mat = matrices[matrix_idx];
        size_t num_rows = static_cast<size_t>(mat.rows());
        size_t num_cols = static_cast<size_t>(mat.cols());
        uint32_t log_num_rows = base::bits::CheckedLog2(num_rows);

        if (reduced_openings[log_num_rows].empty()) {
          reduced_openings[log_num_rows] =
              std::vector<ExtF>(num_rows, ExtF::Zero());
        }
        std::vector<ExtF>& reduced_opening_for_log_num_rows =
            reduced_openings[log_num_rows];
        CHECK_EQ(reduced_opening_for_log_num_rows.size(), num_rows);
        // TODO(ashjeong): Determine if using a matrix is a better fit for
        // |opened_values_for_mat|.
        OpenedValuesForMat opened_values_for_mat = base::CreateVector(
            points[matrix_idx].size(),
            [this, matrix_idx, num_rows, num_cols, log_num_rows, &points, &mat,
             &alpha, &num_reduced, &inv_denoms,
             &reduced_opening_for_log_num_rows](size_t point_idx) {
              const ExtF& point = points[matrix_idx][point_idx];
              math::RowMajorMatrix<F> block =
                  mat.topRows(num_rows >> fri_.log_blowup);
              ReverseMatrixIndexBits(block);
              std::vector<ExtF> ys = InterpolateCoset(
                  block, F::FromMontgomery(F::Config::kSubgroupGenerator),
                  point);
              const ExtF alpha_pow_offset =
                  alpha.Pow(num_reduced[log_num_rows]);
              ExtF alpha_pow = ExtF::One();
              ExtF reduced_ys = ExtF::Zero();
              for (size_t c = 0; c < num_cols - 1; ++c) {
                reduced_ys += alpha_pow * ys[c];
                alpha_pow *= alpha;
              }
              reduced_ys += alpha_pow * ys[num_cols - 1];
              std::vector<ExtF> reduced_rows = DotExtPowers(mat, alpha);
              const std::vector<ExtF>& inv_denom = inv_denoms[point];
              OMP_PARALLEL_FOR(size_t i = 0;
                               i < reduced_opening_for_log_num_rows.size();
                               ++i) {
                reduced_opening_for_log_num_rows[i] +=
                    alpha_pow_offset * (reduced_rows[i] - reduced_ys) *
                    inv_denom[i];
              }
              num_reduced[log_num_rows] += num_cols;
              return ys;
            });
        opened_values_for_round[matrix_idx] = std::move(opened_values_for_mat);
      }
      opened_values[round] = std::move(opened_values_for_round);
    }
    std::vector<std::vector<ExtF>> fri_input;
    fri_input.reserve(reduced_openings.size() - 1);
    for (size_t i = reduced_openings.size() - 1; i != SIZE_MAX; --i) {
      if (!reduced_openings[i].empty()) {
        fri_input.push_back(std::move(reduced_openings[i]));
      }
    }

    *opened_values_out = std::move(opened_values);
    *proof = fri::Prove<TwoAdicFRI>(
        fri_, std::move(fri_input), challenger,
        [this, log_global_max_num_rows, &prover_data_by_round](size_t index) {
          size_t num_rounds = prover_data_by_round.size();
          return base::CreateVector(
              num_rounds, [this, log_global_max_num_rows, index,
                           &prover_data_by_round](size_t round) {
                Proof proof;
                std::vector<std::vector<F>> openings;
                const ProverData& prover_data = prover_data_by_round[round];
                uint32_t log_max_num_rows =
                    base::bits::CheckedLog2(mmcs_.GetMaxRowSize(prover_data));
                uint32_t bits_reduced =
                    log_global_max_num_rows - log_max_num_rows;
                uint32_t reduced_index = index >> bits_reduced;
                CHECK(mmcs_.CreateOpeningProof(reduced_index, prover_data,
                                               &openings, &proof));
                return BatchOpening<TwoAdicFRI>{std::move(openings),
                                                std::move(proof)};
              });
        });
    return true;
  }

  // TODO(ashjeong): remove the use of |std::tuple| and name the separate
  // containers applicable names.
  [[nodiscard]] bool VerifyOpeningProof(
      const std::vector<Commitment>& commits_by_round,
      const std::vector<std::vector<Domain>>& domains_by_round,
      const std::vector<
          std::vector<std::vector<std::tuple<ExtF, std::vector<ExtF>>>>>&
          claims_by_round,
      const FRIProof& proof, Challenger& challenger) {
    TRACE_EVENT("ProofVerification", "TwoAdicFRI::VerifyOpeningProof");
    // Batch combination challenge
    const ExtF alpha = challenger.template SampleExtElement<ExtF>();
    VLOG(2) << "FRI(alpha): " << alpha.ToHexString(true);
    uint32_t log_global_max_num_rows =
        proof.commit_phase_commits.size() + fri_.log_blowup;
    return fri::Verify(
        fri_, proof, challenger,
        [this, alpha, log_global_max_num_rows, &commits_by_round,
         &domains_by_round, &claims_by_round](
            size_t index, const InputProof& input_proof,
            std::vector<size_t>& ro_num_rows, std::vector<ExtF>& ro_values) {
          absl::btree_map<size_t, std::tuple<ExtF, ExtF>> reduced_openings;
          size_t num_rounds = commits_by_round.size();
          for (size_t round = 0; round < num_rounds; ++round) {
            const std::vector<std::vector<std::tuple<ExtF, std::vector<ExtF>>>>&
                claim = claims_by_round[round];
            size_t vals_size = claim.size();
            size_t batch_max_num_rows = 0;
            std::vector<math::Dimensions> batch_dims = base::CreateVector(
                vals_size, [this, round, &batch_max_num_rows,
                            &domains_by_round](size_t batch_idx) {
                  const Domain& mat_domain = domains_by_round[round][batch_idx];
                  size_t num_rows = mat_domain.domain()->size()
                                    << fri_.log_blowup;
                  batch_max_num_rows = std::max(batch_max_num_rows, num_rows);
                  return math::Dimensions{0, num_rows};
                });
            uint32_t bits_reduced = log_global_max_num_rows -
                                    base::bits::CheckedLog2(batch_max_num_rows);
            uint32_t reduced_index = index >> bits_reduced;
            const std::vector<std::vector<F>>& opened_values =
                input_proof[round].opened_values;

            CHECK(mmcs_.VerifyOpeningProof(commits_by_round[round], batch_dims,
                                           reduced_index, opened_values,
                                           input_proof[round].opening_proof));

            for (size_t batch_idx = 0; batch_idx < vals_size; ++batch_idx) {
              const Domain& mat_domain = domains_by_round[round][batch_idx];
              const std::vector<std::tuple<ExtF, std::vector<ExtF>>>&
                  mat_points_and_values = claim[batch_idx];
              uint32_t log_num_rows =
                  mat_domain.domain()->log_size_of_group() + fri_.log_blowup;
              uint32_t bits_reduced = log_global_max_num_rows - log_num_rows;
              uint32_t rev_reduced_index = base::bits::ReverseBitsLen(
                  index >> bits_reduced, log_num_rows);
              F w;
              CHECK(F::GetRootOfUnity(size_t{1} << log_num_rows, &w));
              F x = F::FromMontgomery(F::Config::kSubgroupGenerator) *
                    w.Pow(rev_reduced_index);

              reduced_openings.try_emplace(
                  log_num_rows, std::make_tuple(ExtF::One(), ExtF::Zero()));
              for (size_t i = 0; i < mat_points_and_values.size(); ++i) {
                const ExtF& z = std::get<0>(mat_points_and_values[i]);
                const std::vector<ExtF>& ps_at_z =
                    std::get<1>(mat_points_and_values[i]);
                CHECK_EQ(ps_at_z.size(), opened_values[i].size());
                for (size_t j = 0; j < ps_at_z.size(); ++j) {
                  const ExtF& p_at_z = ps_at_z[j];
                  ExtF quotient =
                      unwrap((ExtF(opened_values[batch_idx][j]) - p_at_z) /
                             (ExtF(x) - z));
                  std::tuple<ExtF, ExtF>& reduced_opening =
                      reduced_openings[log_num_rows];
                  std::get<1>(reduced_opening) +=
                      std::get<0>(reduced_opening) * quotient;
                  std::get<0>(reduced_opening) *= alpha;
                }
              }
            }
          }
          ro_num_rows.reserve(reduced_openings.size());
          ro_values.reserve(reduced_openings.size());
          for (auto it = reduced_openings.rbegin();
               it != reduced_openings.rend(); ++it) {
            ro_num_rows.emplace_back(it->first);
            ro_values.emplace_back(std::move(std::get<1>(it->second)));
          }
        });
  }

 private:
  friend class c::crypto::TwoAdicFRIImpl<ExtF, InputMMCS, ChallengeMMCS,
                                         Challenger>;

  static absl::flat_hash_map<ExtF, std::vector<ExtF>>
  ComputeInverseDenominators(
      const std::vector<absl::Span<const math::RowMajorMatrix<F>>>&
          matrices_by_round,
      const OpeningPoints& points_by_round, F coset_shift) {
    TRACE_EVENT("Utils", "ComputeInverseDenominators");
    size_t num_rounds = matrices_by_round.size();

    absl::flat_hash_map<ExtF, uint32_t> max_log_num_rows_for_point;
    uint32_t max_log_num_rows = 0;
    for (size_t round = 0; round < num_rounds; ++round) {
      absl::Span<const math::RowMajorMatrix<F>> matrices =
          matrices_by_round[round];
      const OpeningPointsForRound& points = points_by_round[round];
      for (const math::RowMajorMatrix<F>& matrix : matrices) {
        uint32_t log_num_rows =
            base::bits::CheckedLog2(static_cast<uint32_t>(matrix.rows()));
        max_log_num_rows = std::max(max_log_num_rows, log_num_rows);
        for (const std::vector<ExtF>& point_list : points) {
          for (const ExtF& point : point_list) {
            const auto [it, inserted] =
                max_log_num_rows_for_point.try_emplace(point, log_num_rows);
            if (!inserted) {
              it->second = std::max(it->second, log_num_rows);
            }
          }
        }
      }
    }

    // Compute the largest subgroup we will use, in bitrev order.
    F w;
    CHECK(F::GetRootOfUnity(size_t{1} << max_log_num_rows, &w));
    // TODO(chokobole): Change type of |subgroup| to |std::vector<ExtF>|.
    std::vector<F> subgroup = F::GetBitRevIndexSuccessivePowers(
        size_t{1} << max_log_num_rows, w, coset_shift);

    absl::flat_hash_map<ExtF, std::vector<ExtF>> ret;
    ret.reserve(max_log_num_rows_for_point.size());
    for (auto it = max_log_num_rows_for_point.begin();
         it != max_log_num_rows_for_point.end(); ++it) {
      const ExtF& point = it->first;
      uint32_t log_num_rows = it->second;
      std::vector<ExtF> temp(size_t{1} << log_num_rows);
      base::Parallelize(
          temp, [&subgroup, &point](absl::Span<ExtF> chunk, size_t chunk_offset,
                                    size_t chunk_size) {
            size_t start = chunk_offset * chunk_size;
            for (size_t i = start; i < start + chunk.size(); ++i) {
              chunk[i - start] = ExtF(subgroup[i]) - point;
            }
            CHECK(ExtF::BatchInverseInPlace(chunk));
          });
      ret[point] = std::move(temp);
    }
    return ret;
  }

  // Slight variation of this approach:
  // https://hackmd.io/@vbuterin/barycentric_evaluation
  template <typename Derived>
  static std::vector<ExtF> InterpolateCoset(
      const Eigen::MatrixBase<Derived>& coset_evals, F shift,
      const ExtF& point) {
    TRACE_EVENT("Utils", "InterpolateCoset");
    size_t num_rows = static_cast<size_t>(coset_evals.rows());
    size_t num_cols = static_cast<size_t>(coset_evals.cols());
    uint32_t log_num_rows = base::bits::CheckedLog2(num_rows);
    F w;
    CHECK(F::GetRootOfUnity(num_rows, &w));

    std::vector<std::vector<ExtF>> sums = base::ParallelizeMap(
        num_rows,
        [num_cols, &w, &point, &shift, &coset_evals](
            size_t chunk_actual_size, size_t chunk_offset, size_t chunk_size) {
          size_t row_start = chunk_offset * chunk_size;
          F pow = w.Pow(row_start);
          std::vector<ExtF> sum_tracker(num_cols, ExtF::Zero());
          ExtF shifted_pow(shift * pow);
          std::vector<ExtF> diff_invs = base::CreateVector(
              chunk_actual_size, [&point, &shifted_pow, &w](size_t i) {
                ExtF temp = point - shifted_pow;
                shifted_pow *= w;
                return temp;
              });
          CHECK(ExtF::BatchInverseInPlaceSerial(diff_invs));
          for (size_t r = 0; r < chunk_actual_size; ++r) {
            for (size_t c = 0; c < num_cols; ++c) {
              sum_tracker[c] +=
                  diff_invs[r] * pow * coset_evals(row_start + r, c);
            }
            pow *= w;
          }
          return sum_tracker;
        });
    const ExtF zeroifier =
        point.ExpPowOfTwo(log_num_rows) - ExtF(shift).ExpPowOfTwo(log_num_rows);
    const F denominator = F(num_rows) * shift.Pow(num_rows - 1);
    const ExtF scale = zeroifier * ExtF(unwrap(denominator.Inverse()));

    std::vector<ExtF> sum(num_cols, ExtF::Zero());
    for (size_t chunk_offset = 0; chunk_offset < sums.size(); ++chunk_offset) {
      for (size_t c = 0; c < num_cols; ++c) {
        sum[c] += sums[chunk_offset][c];
      }
    }
    for (size_t c = 0; c < num_cols; ++c) {
      sum[c] *= scale;
    }
    return sum;
  }

  InputMMCS mmcs_;
  FRIConfig<ChallengeMMCS> fri_;
};

}  // namespace crypto
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_COMMITMENTS_FRI_TWO_ADIC_FRI_H_
