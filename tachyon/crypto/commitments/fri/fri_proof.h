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
#include "tachyon/base/json/json.h"
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

template <typename W, size_t DIGEST_ELEMS>
struct SP1Value {
  std::array<W, DIGEST_ELEMS> value;
};

template <typename PCS>
struct SP1FRIProof {
  FRIProof<PCS> fri_proof;
  std::vector<BatchOpening<PCS>> query_openings;
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

template <typename T, size_t N>
class RapidJsonValueConverter<crypto::SP1Value<T, N>> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(const crypto::SP1Value<T, N>& value,
                               Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    NOTREACHED();
    return object;
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 crypto::SP1Value<T, N>* hash, std::string* error) {
    std::array<T, N> value;
    if (!ParseJsonElement(json_value, "value", &value, error)) return false;
    *hash = {std::move(value)};
    return true;
  }
};

template <typename PCS>
class RapidJsonValueConverter<crypto::BatchOpening<PCS>> {
  using InputMMCS = typename PCS::InputMMCS;
  using Field = typename InputMMCS::Field;
  using Proof = typename InputMMCS::Proof;

 public:
  template <typename Allocator>
  static rapidjson::Value From(const crypto::BatchOpening<PCS>& value,
                               Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    NOTREACHED();
    return object;
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 crypto::BatchOpening<PCS>* value, std::string* error) {
    std::vector<std::vector<Field>> opened_values;
    Proof opening_proof;
    if (!ParseJsonElement(json_value, "opened_values", &opened_values, error))
      return false;
    if (!ParseJsonElement(json_value, "opening_proof", &opening_proof, error))
      return false;
    *value = {std::move(opened_values), std::move(opening_proof)};
    return true;
  }
};

template <typename PCS>
class RapidJsonValueConverter<crypto::CommitPhaseProofStep<PCS>> {
  using InputMMCS = typename PCS::InputMMCS;
  using ChallengeMMCS = typename PCS::ChallengeMMCS;
  using Commitment = typename ChallengeMMCS::Commitment;
  using ExtField = typename ChallengeMMCS::Field;
  using Field = typename InputMMCS::Field;
  using Proof = typename ChallengeMMCS::Proof;

 public:
  template <typename Allocator>
  static rapidjson::Value From(const crypto::CommitPhaseProofStep<PCS>& value,
                               Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    NOTREACHED();
    return object;
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 crypto::CommitPhaseProofStep<PCS>* value, std::string* error) {
    crypto::SP1Value<Field, 4> sibling_value;
    Proof opening_proof;
    if (!ParseJsonElement(json_value, "sibling_value", &sibling_value, error))
      return false;
    if (!ParseJsonElement(json_value, "opening_proof", &opening_proof, error))
      return false;
    value->sibling_value = ExtField::FromBasePrimeFields(sibling_value.value);
    value->opening_proof = opening_proof;
    return true;
  }
};

template <typename PCS>
class RapidJsonValueConverter<crypto::QueryProof<PCS>> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(const crypto::QueryProof<PCS>& value,
                               Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    NOTREACHED();
    return object;
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 crypto::QueryProof<PCS>* value, std::string* error) {
    std::vector<crypto::CommitPhaseProofStep<PCS>> commit_phase_openings;
    if (!ParseJsonElement(json_value, "commit_phase_openings",
                          &commit_phase_openings, error))
      return false;
    value->commit_phase_openings = std::move(commit_phase_openings);
    return true;
  }
};

template <typename PCS>
class RapidJsonValueConverter<crypto::FRIProof<PCS>> {
  using ChallengeMMCS = typename PCS::ChallengeMMCS;
  using Commitment = typename ChallengeMMCS::Commitment;
  using ExtField = typename ChallengeMMCS::Field;
  using InputMMCS = typename PCS::InputMMCS;
  using Field = typename InputMMCS::Field;

 public:
  template <typename Allocator>
  static rapidjson::Value From(const crypto::FRIProof<PCS>& value,
                               Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    NOTREACHED();
    return object;
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 crypto::FRIProof<PCS>* value, std::string* error) {
    std::vector<crypto::SP1Value<Field, 8>> commit_phase_commits;
    std::vector<crypto::QueryProof<PCS>> query_proofs;
    crypto::SP1Value<Field, 4> final_poly;
    Field pow_witness;
    if (!ParseJsonElement(json_value, "commit_phase_commits",
                          &commit_phase_commits, error))
      return false;
    if (!ParseJsonElement(json_value, "query_proofs", &query_proofs, error))
      return false;
    if (!ParseJsonElement(json_value, "final_poly", &final_poly, error))
      return false;
    if (!ParseJsonElement(json_value, "pow_witness", &pow_witness, error))
      return false;
    value->commit_phase_commits.resize(commit_phase_commits.size());
    for (size_t i = 0; i < value->commit_phase_commits.size(); ++i) {
      value->commit_phase_commits[i] = std::move(commit_phase_commits[i].value);
    }
    value->query_proofs = std::move(query_proofs);
    value->final_eval = ExtField::FromBasePrimeFields(final_poly.value);
    value->pow_witness = std::move(pow_witness);
    // *value = {std::move(commit_phase_commits), std::move(opening_proof)};
    return true;
  }
};

template <typename PCS>
class RapidJsonValueConverter<crypto::SP1FRIProof<PCS>> {
  using InputProof = typename PCS::InputProof;

  InputProof input_proof;

 public:
  template <typename Allocator>
  static rapidjson::Value From(const crypto::SP1FRIProof<PCS>& value,
                               Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    NOTREACHED();
    return object;
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 crypto::SP1FRIProof<PCS>* value, std::string* error) {
    crypto::FRIProof<PCS> fri_proof;
    std::vector<InputProof> query_openings;
    if (!ParseJsonElement(json_value, "fri_proof", &fri_proof, error))
      return false;
    if (!ParseJsonElement(json_value, "query_openings", &query_openings, error))
      return false;
    value->fri_proof = std::move(fri_proof);
    CHECK_EQ(query_openings.size(), value->fri_proof.query_proofs.size());
    for (size_t i = 0; i < query_openings.size(); ++i) {
      value->fri_proof.query_proofs[i].input_proof = query_openings[i];
    }
    return true;
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_COMMITMENTS_FRI_FRI_PROOF_H_
