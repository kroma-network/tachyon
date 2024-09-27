// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_CRYPTO_COMMITMENTS_FRI_FRI_PROOF_H_
#define TACHYON_CRYPTO_COMMITMENTS_FRI_FRI_PROOF_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/crypto/commitments/mixed_matrix_commitment_scheme_traits_forward.h"

namespace tachyon {
namespace crypto {

template <typename PCS>
struct BatchOpening {
  using InputMMCS = typename PCS::InputMMCS;
  using Field = typename InputMMCS::Field;
  using Proof = typename InputMMCS::Proof;

  std::vector<std::vector<Field>> opened_values;
  Proof opening_proof;

  bool operator==(const BatchOpening& other) const {
    return opened_values == other.opened_values &&
           opening_proof == other.opening_proof;
  }
  bool operator!=(const BatchOpening& other) const {
    return !operator==(other);
  }

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

  bool operator==(const CommitPhaseResult& other) const {
    return commits == other.commits && data == other.data &&
           final_eval == other.final_eval;
  }
  bool operator!=(const CommitPhaseResult& other) const {
    return !operator==(other);
  }

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

  bool operator==(const CommitPhaseProofStep& other) const {
    return sibling_value == other.sibling_value &&
           opening_proof == other.opening_proof;
  }
  bool operator!=(const CommitPhaseProofStep& other) const {
    return !operator==(other);
  }

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

  bool operator==(const QueryProof& other) const {
    return input_proof == other.input_proof &&
           commit_phase_openings == other.commit_phase_openings;
  }
  bool operator!=(const QueryProof& other) const { return !operator==(other); }

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

  bool operator==(const FRIProof& other) const {
    return commit_phase_commits == other.commit_phase_commits &&
           query_proofs == other.query_proofs &&
           final_eval == other.final_eval && pow_witness == other.pow_witness;
  }
  bool operator!=(const FRIProof& other) const { return !operator==(other); }

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

}  // namespace crypto

namespace base {

template <typename PCS>
class Copyable<crypto::BatchOpening<PCS>> {
  using InputMMCS = typename PCS::InputMMCS;
  using Field = typename InputMMCS::Field;
  using Proof = typename InputMMCS::Proof;

 public:
  static bool WriteTo(const crypto::BatchOpening<PCS>& batch_opening,
                      Buffer* buffer) {
    return buffer->WriteMany(batch_opening.opened_values,
                             batch_opening.opening_proof);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       crypto::BatchOpening<PCS>* batch_opening) {
    std::vector<std::vector<Field>> opened_values;
    Proof opening_proof;
    if (!buffer.ReadMany(&opened_values, &opening_proof)) {
      return false;
    }

    *batch_opening = {std::move(opened_values), std::move(opening_proof)};
    return true;
  }

  static size_t EstimateSize(const crypto::BatchOpening<PCS>& batch_opening) {
    return base::EstimateSize(batch_opening.opened_values,
                              batch_opening.opening_proof);
  }
};

template <typename PCS>
class Copyable<crypto::CommitPhaseResult<PCS>> {
  using ChallengeMMCS = typename PCS::ChallengeMMCS;
  using Commitment = typename ChallengeMMCS::Commitment;
  using ProverData = typename ChallengeMMCS::ProverData;
  using Field = typename ChallengeMMCS::Field;

 public:
  static bool WriteTo(const crypto::CommitPhaseResult<PCS>& commit_phase_result,
                      Buffer* buffer) {
    return buffer->WriteMany(commit_phase_result.commits,
                             commit_phase_result.data,
                             commit_phase_result.final_eval);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       crypto::CommitPhaseResult<PCS>* commit_phase_result) {
    std::vector<Commitment> commits;
    std::vector<ProverData> data;
    Field final_eval;
    if (!buffer.ReadMany(&commits, &data, &final_eval)) {
      return false;
    }

    *commit_phase_result = {std::move(commits), std::move(data),
                            std::move(final_eval)};
    return true;
  }

  static size_t EstimateSize(
      const crypto::CommitPhaseResult<PCS>& commit_phase_result) {
    return base::EstimateSize(commit_phase_result.commits,
                              commit_phase_result.data,
                              commit_phase_result.final_eval);
  }
};

template <typename PCS>
class Copyable<crypto::CommitPhaseProofStep<PCS>> {
  using ChallengeMMCS = typename PCS::ChallengeMMCS;
  using Field = typename ChallengeMMCS::Field;
  using Proof = typename ChallengeMMCS::Proof;

 public:
  static bool WriteTo(
      const crypto::CommitPhaseProofStep<PCS>& commit_phase_proof_step,
      Buffer* buffer) {
    return buffer->WriteMany(commit_phase_proof_step.sibling_value,
                             commit_phase_proof_step.opening_proof);
  }

  static bool ReadFrom(
      const ReadOnlyBuffer& buffer,
      crypto::CommitPhaseProofStep<PCS>* commit_phase_proof_step) {
    Field sibling_value;
    Proof opening_proof;
    if (!buffer.ReadMany(&sibling_value, &opening_proof)) {
      return false;
    }

    *commit_phase_proof_step = {std::move(sibling_value),
                                std::move(opening_proof)};
    return true;
  }

  static size_t EstimateSize(
      const crypto::CommitPhaseProofStep<PCS>& commit_phase_proof_step) {
    return base::EstimateSize(commit_phase_proof_step.sibling_value,
                              commit_phase_proof_step.opening_proof);
  }
};

template <typename PCS>
class Copyable<crypto::QueryProof<PCS>> {
  using InputProof = typename PCS::InputProof;

 public:
  static bool WriteTo(const crypto::QueryProof<PCS>& query_proof,
                      Buffer* buffer) {
    return buffer->WriteMany(query_proof.input_proof,
                             query_proof.commit_phase_openings);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       crypto::QueryProof<PCS>* query_proof) {
    InputProof input_proof;
    std::vector<crypto::CommitPhaseProofStep<PCS>> commit_phase_openings;
    if (!buffer.ReadMany(&input_proof, &commit_phase_openings)) {
      return false;
    }

    *query_proof = {std::move(input_proof), std::move(commit_phase_openings)};
    return true;
  }

  static size_t EstimateSize(const crypto::QueryProof<PCS>& query_proof) {
    return base::EstimateSize(query_proof.input_proof,
                              query_proof.commit_phase_openings);
  }
};

template <typename PCS>
class Copyable<crypto::FRIProof<PCS>> {
  using ChallengeMMCS = typename PCS::ChallengeMMCS;
  using Commitment = typename ChallengeMMCS::Commitment;
  using ExtField = typename ChallengeMMCS::Field;
  using InputMMCS = typename PCS::InputMMCS;
  using Field = typename InputMMCS::Field;

 public:
  static bool WriteTo(const crypto::FRIProof<PCS>& fri_proof, Buffer* buffer) {
    return buffer->WriteMany(fri_proof.commit_phase_commits,
                             fri_proof.query_proofs, fri_proof.final_eval,
                             fri_proof.pow_witness);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       crypto::FRIProof<PCS>* fri_proof) {
    std::vector<Commitment> commit_phase_commits;
    std::vector<crypto::QueryProof<PCS>> query_proofs;
    ExtField final_eval;
    Field pow_witness;
    if (!buffer.ReadMany(&commit_phase_commits, &query_proofs, &final_eval,
                         &pow_witness)) {
      return false;
    }

    *fri_proof = {std::move(commit_phase_commits), std::move(query_proofs),
                  std::move(final_eval), std::move(pow_witness)};
    return true;
  }

  static size_t EstimateSize(const crypto::FRIProof<PCS>& fri_proof) {
    return base::EstimateSize(fri_proof.commit_phase_commits,
                              fri_proof.query_proofs, fri_proof.final_eval,
                              fri_proof.pow_witness);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_COMMITMENTS_FRI_FRI_PROOF_H_
