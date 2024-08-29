// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_CRYPTO_COMMITMENTS_FRI_FRI_PROOF_H_
#define TACHYON_CRYPTO_COMMITMENTS_FRI_FRI_PROOF_H_

#include <vector>

#include "tachyon/crypto/commitments/mixed_matrix_commitment_scheme_traits_forward.h"

namespace tachyon::crypto {

template <typename PCS>
struct BatchOpening {
  using InputMMCS = typename PCS::InputMMCS;
  using Field = typename InputMMCS::Field;
  using Proof = typename InputMMCS::Proof;

  std::vector<std::vector<Field>> opened_values;
  Proof opening_proof;
};

template <typename PCS>
struct CommitPhaseResult {
  using ChallengeMMCS = typename PCS::ChallengeMMCS;
  using Commitment = typename ChallengeMMCS::Commitment;
  using ProverData = typename ChallengeMMCS::ProverData;
  using Field = typename ChallengeMMCS::Field;

  std::vector<Commitment> commits;
  std::vector<ProverData> data;
  Field final_eval;
};

template <typename PCS>
struct CommitPhaseProofStep {
  using ChallengeMMCS = typename PCS::ChallengeMMCS;
  using Field = typename ChallengeMMCS::Field;
  using Proof = typename ChallengeMMCS::Proof;

  // The opening of the commit phase codeword at the sibling location.
  Field sibling_value;
  Proof opening_proof;
};

// Note(ashjeong): |InputProof| is usually a vector of |BatchOpening|
template <typename PCS>
struct QueryProof {
  using InputProof = typename PCS::InputProof;

  InputProof input_proof;
  // For each commit phase commitment, this contains openings of a commit phase
  // codeword at the queried location, along with an opening proof.
  std::vector<CommitPhaseProofStep<PCS>> commit_phase_openings;
};

template <typename PCS>
struct FriProof {
  using ChallengeMMCS = typename PCS::ChallengeMMCS;
  using Commitment = typename ChallengeMMCS::Commitment;
  using ExtField = typename ChallengeMMCS::Field;
  using InputMMCS = typename PCS::InputMMCS;
  using Field = typename InputMMCS::Field;

  std::vector<Commitment> commit_phase_commits;
  std::vector<QueryProof<PCS>> query_proofs;
  ExtField final_eval;
  Field pow_witness;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_FRI_FRI_PROOF_H_
