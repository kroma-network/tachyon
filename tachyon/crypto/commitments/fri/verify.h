// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_CRYPTO_COMMITMENTS_FRI_VERIFY_H_
#define TACHYON_CRYPTO_COMMITMENTS_FRI_VERIFY_H_

#include <vector>

#include "tachyon/base/bits.h"
#include "tachyon/base/ranges/algorithm.h"
#include "tachyon/crypto/challenger/challenger.h"
#include "tachyon/crypto/commitments/fri/fri_config.h"
#include "tachyon/crypto/commitments/fri/fri_proof.h"
#include "tachyon/math/geometry/dimensions.h"

namespace tachyon::crypto::fri {

template <typename PCS>
struct CommitStep {
  using ChallengeMMCS = typename PCS::ChallengeMMCS;
  using F = typename ChallengeMMCS::Field;
  using Commitment = typename ChallengeMMCS::Commitment;

  F beta;
  Commitment commit;
  CommitPhaseProofStep<PCS> opening;
};

template <typename PCS, typename MMCS, typename F>
F VerifyQuery(uint32_t index, uint32_t log_max_num_rows,
              const FRIConfig<MMCS>& config,
              const std::vector<CommitStep<PCS>>& steps,
              const std::vector<size_t>& ro_num_rows,
              const std::vector<F>& ro_values) {
  F folded_eval = F::Zero();
  size_t ro_idx = 0;
  size_t ro_size = ro_num_rows.size();

  for (uint32_t step_idx = 0; step_idx < steps.size(); ++step_idx) {
    uint32_t log_folded_num_rows = log_max_num_rows - step_idx - 1;
    if (ro_idx != ro_size && ro_num_rows[ro_idx] == log_folded_num_rows + 1) {
      folded_eval += ro_values[ro_idx++];
    }
    size_t index_sibling = index ^ 1;
    size_t index_pair = index >> 1;
    std::vector<std::vector<F>> evals = {{folded_eval, folded_eval}};
    evals[0][index_sibling % 2] = steps[step_idx].opening.sibling_value;
    CHECK(config.mmcs.VerifyOpeningProof(
        steps[step_idx].commit,
        {math::Dimensions(2, size_t{1} << log_folded_num_rows)}, index_pair,
        evals, steps[step_idx].opening.opening_proof));
    folded_eval = FoldRow(index_pair, log_folded_num_rows, steps[step_idx].beta,
                          evals[0]);
    index = index_pair;
  }
  CHECK_LT(index, config.Blowup()) << "index was " << index;
  CHECK_EQ(ro_idx, ro_size);

  return folded_eval;
}

template <typename PCS, typename ChallengeMMCS, typename Challenger,
          typename OpenInputCallback>
[[nodiscard]] bool Verify(const FRIConfig<ChallengeMMCS>& config,
                          const FRIProof<PCS>& proof, Challenger& challenger,
                          OpenInputCallback open_input) {
  using ExtF = typename ChallengeMMCS::Field;
  using Commitment = typename ChallengeMMCS::Commitment;
  size_t num_commits = proof.commit_phase_commits.size();
  std::vector<ExtF> betas = base::Map(
      proof.commit_phase_commits,
      [&challenger](size_t i, const Commitment& commit) {
        challenger.ObserveContainer(commit);
        ExtF beta = challenger.template SampleExtElement<ExtF>();
        VLOG(2) << "FRI(beta[" << i << "]): " << beta.ToHexString(true);
        return beta;
      });
  challenger.ObserveContainer(proof.final_eval);
  VLOG(2) << "FRI(final_eval): " << proof.final_eval.ToHexString(true);

  if (proof.query_proofs.size() != config.num_queries) {
    LOG(ERROR) << "proof size doesn't match " << proof.query_proofs.size()
               << " vs " << config.num_queries;
    return false;
  }
  // Check PoW.
  VLOG(2) << "FRI(pow): " << proof.pow_witness.ToHexString(true);
  if (!(challenger.CheckWitness(config.proof_of_work_bits,
                                proof.pow_witness))) {
    LOG(ERROR) << "failed to check pow";
    return false;
  }

  uint32_t log_max_num_rows = num_commits + config.log_blowup;

  for (size_t i = 0; i < proof.query_proofs.size(); ++i) {
    std::vector<size_t> ro_num_rows;
    std::vector<ExtF> ro_values;
    size_t index = challenger.SampleBits(log_max_num_rows);
    VLOG(2) << "FRI(index[" << i << "]): " << index;
    open_input(index, proof.query_proofs[i].input_proof, ro_num_rows,
               ro_values);

#if DCHECK_IS_ON()
    // Check reduced openings sorted by |num_rows| descending
    DCHECK(base::ranges::is_sorted(ro_num_rows.begin(), ro_num_rows.end(),
                                   base::ranges::greater()));
#endif
    std::vector<CommitStep<PCS>> steps =
        base::CreateVector(num_commits, [&betas, &proof, i](size_t j) {
          return CommitStep<PCS>{
              betas[j], proof.commit_phase_commits[j],
              proof.query_proofs[i].commit_phase_openings[j]};
        });
    ExtF folded_eval = VerifyQuery(index, log_max_num_rows, config, steps,
                                   ro_num_rows, ro_values);
    if (folded_eval != proof.final_eval) {
      LOG(ERROR) << "final_eval is not matched: "
                 << folded_eval.ToHexString(true);
      return false;
    }
  }
  return true;
}

}  // namespace tachyon::crypto::fri

#endif  // TACHYON_CRYPTO_COMMITMENTS_FRI_VERIFY_H_
