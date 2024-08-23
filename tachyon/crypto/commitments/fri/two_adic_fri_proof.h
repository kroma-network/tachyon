// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_CRYPTO_COMMITMENTS_FRI_TWO_ADIC_FRI_PROOF_H_
#define TACHYON_CRYPTO_COMMITMENTS_FRI_TWO_ADIC_FRI_PROOF_H_

#include <vector>

#include "tachyon/crypto/commitments/mixed_matrix_commitment_scheme_traits_forward.h"

namespace tachyon::crypto {

template <typename MMCS>
struct BatchOpening {
  std::vector<std::vector<typename MMCS::Field>> opened_values;
  typename MMCS::Proof opening_proof;
};

template <typename MMCS>
struct CommitPhaseResult {
  std::vector<typename MMCS::Commitment> commits;
  std::vector<typename MMCS::ProverData> data;
  typename MMCS::Field final_eval;
};

template <typename MMCS>
struct CommitPhaseProofStep {
  // The opening of the commit phase codeword at the sibling location.
  typename MMCS::Field sibling_value;
  typename MMCS::Proof opening_proof;
};

// Note(ashjeong): |InputProof| is usually a vector of |BatchOpening|
template <typename MMCS, typename InputProof>
struct QueryProof {
  InputProof input_proof;
  // For each commit phase commitment, this contains openings of a commit phase
  // codeword at the queried location, along with an opening proof.
  std::vector<CommitPhaseProofStep<MMCS>> commit_phase_openings;
};

template <typename MMCS, typename InputProof, typename Witness>
struct TwoAdicFriProof {
  std::vector<typename MMCS::Commitment> commit_phase_commits;
  std::vector<QueryProof<MMCS, InputProof>> query_proofs;
  typename MMCS::Field final_eval;
  Witness pow_witness;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_FRI_TWO_ADIC_FRI_PROOF_H_
