// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_CRYPTO_COMMITMENTS_FRI_TWO_ADIC_FRI_PROVER_H_
#define TACHYON_CRYPTO_COMMITMENTS_FRI_TWO_ADIC_FRI_PROVER_H_

#include <utility>
#include <vector>

#include "tachyon/base/bits.h"
#include "tachyon/base/ranges/algorithm.h"
#include "tachyon/crypto/commitments/fri/fri_config.h"
#include "tachyon/crypto/commitments/fri/two_adic_fri_proof.h"
#include "tachyon/math/finite_fields/extension_field_traits_forward.h"
#include "tachyon/math/matrix/matrix_types.h"
#include "tachyon/zk/air/plonky3/challenger/challenger.h"

namespace tachyon::crypto {

template <typename PCS, typename ExtF, typename ChallengeMMCS,
          typename Challenger>
CommitPhaseResult<PCS> CommitPhase(FriConfig<ChallengeMMCS>& config,
                                   std::vector<std::vector<ExtF>>&& inputs,
                                   Challenger& challenger) {
  // NOTE(ashjeong): This is empirically determined in case the size of the for
  // loop is small enough to not need parallelization.
  constexpr size_t kThreshold = 1024;

  using Commitment = typename ChallengeMMCS::Commitment;
  using ProverData = typename ChallengeMMCS::ProverData;

  // NOTE(ashjeong): Refer to the note below as to why this is safe.
  std::vector<ExtF> folded = std::move(inputs[0]);
  size_t total_folds = base::bits::CheckedLog2(folded.size()) -
                       base::bits::Log2Floor(config.Blowup());
  std::vector<Commitment> commits;
  commits.reserve(total_folds);
  std::vector<ProverData> data;
  data.reserve(total_folds);

  while (folded.size() > config.Blowup()) {
    math::RowMajorMatrix<ExtF> leaves((folded.size() + 1) / 2, 2);
    base::Parallelize(
        folded,
        [&leaves](absl::Span<ExtF> chunk, size_t chunk_offset,
                  size_t chunk_size) {
          size_t chunk_start = chunk_offset * chunk_size;
          for (size_t i = chunk_start; i < chunk_start + chunk.size(); ++i) {
            leaves(i / 2, i % 2) = std::move(chunk[i - chunk_start]);
          }
        },
        kThreshold);

    Commitment commit;
    ProverData prover_data;
    CHECK(config.mmcs.Commit({std::move(leaves)}, &commit, &prover_data));
    commits.push_back(std::move(commit));
    data.push_back(std::move(prover_data));

    challenger.ObserveContainer(commits.back());
    const ExtF beta = challenger.template SampleExtElement<ExtF>();
    VLOG(2) << "FRI(beta[" << commits.size() - 1
            << "]): " << beta.ToHexString(true);

    folded = FoldMatrix(beta, config.mmcs.GetMatrices(data.back()).back());
    // NOTE(ashjeong): |inputs| is sorted by largest to smallest size, and the
    // size of |folded| is divided by two every loop. This means that if
    // |folded| is initialized as |inputs[0]|, the size of the next iteration of
    // |folded| will never be the size of |inputs[0]|.
    for (size_t i = 1; i < inputs.size(); ++i) {
      if (inputs[i].size() == folded.size()) {
        OMP_PARALLEL_FOR(size_t j = 0; j < inputs[i].size(); ++j) {
          folded[j] += inputs[i][j];
        }
      }
    }
  }
  CHECK_EQ(folded.size(), config.Blowup());
  ExtF& final_eval = folded[0];
  VLOG(2) << "FRI(final_eval): " << final_eval.ToHexString(true);

#if DCHECK_IS_ON()
  OMP_PARALLEL_FOR(size_t i = 0; i < folded.size(); ++i) {
    DCHECK_EQ(folded[i], final_eval);
  }
#endif
  challenger.ObserveContainer(final_eval);

  return {std::move(commits), std::move(data), std::move(final_eval)};
}

template <typename PCS, typename ChallengeMMCS = typename PCS::ChallengeMMCS>
std::vector<CommitPhaseProofStep<PCS>> AnswerQuery(
    size_t index, FriConfig<ChallengeMMCS>& config,
    const std::vector<typename ChallengeMMCS::ProverData>&
        commit_phase_commits) {
  return base::CreateVector(
      commit_phase_commits.size(),
      [index, &config, &commit_phase_commits](size_t i) {
        using F = typename ChallengeMMCS::Field;
        using Proof = typename ChallengeMMCS::Proof;

        size_t index_i = index >> i;
        size_t index_i_sibling = index_i ^ 1;
        size_t index_pair = index_i >> 1;
        std::vector<std::vector<F>> opened_rows;
        Proof opening_proof;
        CHECK(config.mmcs.CreateOpeningProof(
            index_pair, commit_phase_commits[i], &opened_rows, &opening_proof));
        CHECK_EQ(opened_rows.size(), size_t{1});
        CHECK_EQ(opened_rows[0].size(), size_t{2});
        const F& sibling_value = opened_rows[0][index_i_sibling % 2];
        return CommitPhaseProofStep<PCS>{sibling_value,
                                         std::move(opening_proof)};
      });
}

template <typename PCS, typename ExtF, typename ChallengeMMCS,
          typename Challenger, typename OpenInputCallback,
          typename F = typename math::ExtensionFieldTraits<ExtF>::BaseField>
TwoAdicFriProof<PCS> TwoAdicFriPCSProve(FriConfig<ChallengeMMCS>& config,
                                        std::vector<std::vector<ExtF>>&& inputs,
                                        Challenger& challenger,
                                        OpenInputCallback open_input) {
  using QueryProof = QueryProof<PCS>;

#if DCHECK_IS_ON()
  // Ensure |inputs| is in order from largest to smallest
  DCHECK(base::ranges::is_sorted(
      inputs.begin(), inputs.end(),
      [](const std::vector<ExtF>& l, const std::vector<ExtF>& r) {
        return l.size() > r.size();
      }));
#endif

  uint32_t log_max_num_rows = base::bits::CheckedLog2(inputs[0].size());
  CommitPhaseResult<PCS> commit_phase_result =
      CommitPhase<PCS>(config, std::move(inputs), challenger);
  F pow_witness = challenger.Grind(config.proof_of_work_bits);
  VLOG(2) << "FRI(pow): " << pow_witness.ToHexString(true);

  std::vector<QueryProof> query_proofs = base::CreateVector(
      config.num_queries, [log_max_num_rows, &challenger, &open_input, &config,
                           &commit_phase_result](size_t query_idx) {
        size_t index = challenger.SampleBits(log_max_num_rows);
        VLOG(2) << "FRI(index[" << query_idx << "]): " << index;
        std::vector<BatchOpening<PCS>> x = open_input(index);

        std::vector<CommitPhaseProofStep<PCS>> answered_query =
            AnswerQuery<PCS>(index, config, commit_phase_result.data);
        return QueryProof{std::move(x), std::move(answered_query)};
      });
  return {std::move(commit_phase_result.commits), std::move(query_proofs),
          std::move(commit_phase_result.final_eval), std::move(pow_witness)};
}

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_FRI_TWO_ADIC_FRI_PROVER_H_
