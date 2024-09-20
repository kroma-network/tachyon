// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_CRYPTO_COMMITMENTS_FRI_FRI_PROOF_H_
#define TACHYON_CRYPTO_COMMITMENTS_FRI_FRI_PROOF_H_

#include <string>
#include <vector>

#include "absl/strings/substitute.h"

#include "tachyon/base/strings/string_util.h"
#include "tachyon/crypto/commitments/mixed_matrix_commitment_scheme_traits_forward.h"

namespace tachyon::crypto {

template <typename PCS>
struct BatchOpening {
  using InputMMCS = typename PCS::InputMMCS;
  using Field = typename InputMMCS::Field;
  using Proof = typename InputMMCS::Proof;

  std::vector<std::vector<Field>> opened_values;
  Proof opening_proof;

  std::string ToString() const {
    return absl::Substitute("{opened_values: $0, opening_proof: $1}",
                            base::Container2DToString(opened_values),
                            base::Container2DToString(opening_proof));
  }

  std::string ToHexString(bool pad_zero = false) const {
    return absl::Substitute(
        "{opened_values: $0, opening_proof: $1}",
        base::Container2DToHexString(opened_values, pad_zero),
        base::Container2DToHexString(opening_proof, pad_zero));
  }
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

  std::string ToString() const {
    return absl::Substitute("{commits: $0, data: $1, final_eval: $2},",
                            base::ContainerToString(commits),
                            base::ContainerToString(data),
                            final_eval.ToString());
  }

  std::string ToHexString(bool pad_zero = false) const {
    return absl::Substitute("{commits: $0, data: $1, final_eval: $2},",
                            base::ContainerToHexString(commits, pad_zero),
                            base::ContainerToHexString(data, pad_zero),
                            final_eval.ToHexString(pad_zero));
  }
};

template <typename PCS>
struct CommitPhaseProofStep {
  using ChallengeMMCS = typename PCS::ChallengeMMCS;
  using Field = typename ChallengeMMCS::Field;
  using Proof = typename ChallengeMMCS::Proof;

  // The opening of the commit phase codeword at the sibling location.
  Field sibling_value;
  Proof opening_proof;

  std::string ToString() const {
    return absl::Substitute("{sibling_value: $0, opening_proof: $1}",
                            sibling_value.ToString(),
                            base::Container2DToString(opening_proof));
  }

  std::string ToHexString(bool pad_zero = false) const {
    return absl::Substitute(
        "{sibling_value: $0, opening_proof: $1}",
        sibling_value.ToHexString(pad_zero),
        base::Container2DToHexString(opening_proof, pad_zero));
  }
};

// Note(ashjeong): |InputProof| is usually a vector of |BatchOpening|
template <typename PCS>
struct QueryProof {
  using InputProof = typename PCS::InputProof;

  InputProof input_proof;
  // For each commit phase commitment, this contains openings of a commit phase
  // codeword at the queried location, along with an opening proof.
  std::vector<CommitPhaseProofStep<PCS>> commit_phase_openings;

  std::string ToString() const {
    return absl::Substitute("{input_proof: $0, commit_phase_openings: $1}",
                            base::ContainerToString(input_proof),
                            base::ContainerToString(commit_phase_openings));
  }

  std::string ToHexString(bool pad_zero = false) const {
    return absl::Substitute("{input_proof: $0, commit_phase_openings: $1}",
                            base::ContainerToHexString(input_proof, pad_zero),
                            base::ContainerToHexString(commit_phase_openings));
  }
};

template <typename PCS>
struct FRIProof {
  using ChallengeMMCS = typename PCS::ChallengeMMCS;
  using Commitment = typename ChallengeMMCS::Commitment;
  using ExtField = typename ChallengeMMCS::Field;
  using InputMMCS = typename PCS::InputMMCS;
  using Field = typename InputMMCS::Field;

  std::vector<Commitment> commit_phase_commits;
  std::vector<QueryProof<PCS>> query_proofs;
  ExtField final_eval;
  Field pow_witness;

  std::string ToString() const {
    return absl::Substitute(
        "{commit_phase_commits: $0, query_proofs: $1, final_eval: $2, "
        "pow_witness: $3}",
        base::Container2DToString(commit_phase_commits),
        base::ContainerToString(query_proofs), final_eval.ToString(),
        pow_witness.ToString());
  }

  std::string ToHexString(bool pad_zero = false) const {
    return absl::Substitute(
        "{commit_phase_commits: $0, query_proofs: $1, final_eval: $2, "
        "pow_witness: $3}",
        base::Container2DToHexString(commit_phase_commits, pad_zero),
        base::ContainerToHexString(query_proofs, pad_zero),
        final_eval.ToHexString(pad_zero), pow_witness.ToHexString(pad_zero));
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_FRI_FRI_PROOF_H_
