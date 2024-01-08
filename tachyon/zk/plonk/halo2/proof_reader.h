#ifndef TACHYON_ZK_PLONK_HALO2_PROOF_READER_H_
#define TACHYON_ZK_PLONK_HALO2_PROOF_READER_H_

#include <utility>
#include <vector>

#include "tachyon/base/logging.h"
#include "tachyon/crypto/transcripts/transcript.h"
#include "tachyon/zk/plonk/halo2/proof.h"
#include "tachyon/zk/plonk/keys/verifying_key.h"
#include "tachyon/zk/plonk/permutation/permutation_utils.h"

namespace tachyon::zk::halo2 {

enum class ProofCursor {
  kAdviceCommitmentsVecAndChallenges,
  kTheta,
  kLookupPermutedCommitments,
  kBetaAndGamma,
  kPermutationProductCommitments,
  kLookupProductCommitments,
  kVanishingRandomPolyCommitment,
  kY,
  kVanishingHPolyCommitments,
  kX,
  kInstanceEvals,
  kAdviceEvals,
  kFixedEvals,
  kVanishingEval,
  kCommonPermutationEvals,
  kPermutationEvals,
  kLookupEvalsVec,
  kDone,
};

template <typename PCS>
class ProofReader {
 public:
  using F = typename PCS::Field;
  using C = typename PCS::Commitment;

  ProofReader(const VerifyingKey<PCS>& verifying_key,
              crypto::TranscriptReader<C>* transcript, size_t num_circuits)
      : verifying_key_(verifying_key),
        transcript_(transcript),
        num_circuits_(num_circuits) {}

  const Proof<F, C>& proof() const { return proof_; }
  Proof<F, C>& proof() { return proof_; }

  void ReadAdviceCommitmentsVecAndChallenges() {
    CHECK_EQ(cursor_, ProofCursor::kAdviceCommitmentsVecAndChallenges);
    const ConstraintSystem<F>& constraint_system =
        verifying_key_.constraint_system();
    proof_.advices_commitments_vec.resize(num_circuits_);
    for (size_t i = 0; i < num_circuits_; ++i) {
      proof_.advices_commitments_vec[i].reserve(
          constraint_system.num_advice_columns());
    }
    proof_.challenges.reserve(constraint_system.challenge_phases().size());
    for (Phase current_phase : constraint_system.GetPhases()) {
      for (size_t i = 0; i < num_circuits_; ++i) {
        for (Phase phase : constraint_system.advice_column_phases()) {
          if (current_phase == phase) {
            proof_.advices_commitments_vec[i].push_back(Read<C>());
          }
        }
      }
      for (Phase phase : constraint_system.challenge_phases()) {
        if (current_phase == phase) {
          proof_.challenges.push_back(transcript_->SqueezeChallenge());
        }
      }
    }
    cursor_ = ProofCursor::kTheta;
  }

  void ReadTheta() {
    CHECK_EQ(cursor_, ProofCursor::kTheta);
    proof_.theta = transcript_->SqueezeChallenge();
    cursor_ = ProofCursor::kLookupPermutedCommitments;
  }

  void ReadLookupPermutedCommitments() {
    CHECK_EQ(cursor_, ProofCursor::kLookupPermutedCommitments);
    size_t num_lookups = verifying_key_.constraint_system().lookups().size();
    proof_.lookup_permuted_commitments_vec =
        base::CreateVector(num_circuits_, [this, num_lookups]() {
          return base::CreateVector(num_lookups, [this]() {
            C input = Read<C>();
            C table = Read<C>();
            return LookupPair<C>(std::move(input), std::move(table));
          });
        });
    cursor_ = ProofCursor::kBetaAndGamma;
  }

  void ReadBetaAndGamma() {
    CHECK_EQ(cursor_, ProofCursor::kBetaAndGamma);
    proof_.beta = transcript_->SqueezeChallenge();
    proof_.gamma = transcript_->SqueezeChallenge();
    cursor_ = ProofCursor::kPermutationProductCommitments;
  }

  void ReadPermutationProductCommitments() {
    CHECK_EQ(cursor_, ProofCursor::kPermutationProductCommitments);
    const ConstraintSystem<F>& constraint_system =
        verifying_key_.constraint_system();
    size_t num_products = constraint_system.ComputePermutationProductNums();
    proof_.permutation_product_commitments_vec = base::CreateVector(
        num_circuits_,
        [this, num_products]() { return ReadMany<C>(num_products); });
    cursor_ = ProofCursor::kLookupProductCommitments;
  }

  void ReadLookupProductCommitments() {
    CHECK_EQ(cursor_, ProofCursor::kLookupProductCommitments);
    size_t num_lookups = verifying_key_.constraint_system().lookups().size();
    proof_.lookup_product_commitments_vec = base::CreateVector(
        num_circuits_,
        [this, num_lookups]() { return ReadMany<C>(num_lookups); });
    cursor_ = ProofCursor::kVanishingRandomPolyCommitment;
  }

  void ReadVanishingRandomPolyCommitment() {
    CHECK_EQ(cursor_, ProofCursor::kVanishingRandomPolyCommitment);
    proof_.vanishing_random_poly_commitment = Read<C>();
    cursor_ = ProofCursor::kY;
  }

  void ReadY() {
    CHECK_EQ(cursor_, ProofCursor::kY);
    proof_.y = transcript_->SqueezeChallenge();
    cursor_ = ProofCursor::kVanishingHPolyCommitments;
  }

  void ReadVanishingHPolyCommitments() {
    CHECK_EQ(cursor_, ProofCursor::kVanishingHPolyCommitments);
    size_t quotient_poly_degree =
        verifying_key_.constraint_system().ComputeDegree() - 1;
    proof_.vanishing_h_poly_commitments = ReadMany<C>(quotient_poly_degree);
    cursor_ = ProofCursor::kX;
  }

  void ReadX() {
    CHECK_EQ(cursor_, ProofCursor::kX);
    proof_.x = transcript_->SqueezeChallenge();
    cursor_ = ProofCursor::kInstanceEvals;
  }

  void ReadInstanceEvalsIfQueryInstance() {
    CHECK_EQ(cursor_, ProofCursor::kInstanceEvals);
    size_t num_instance_queries =
        verifying_key_.constraint_system().instance_queries().size();
    return base::CreateVector(num_circuits_, [this, num_instance_queries]() {
      return ReadMany<F>(num_instance_queries);
    });
    cursor_ = ProofCursor::kAdviceEvals;
  }

  void ReadInstanceEvalsIfNoQueryInstance() {
    CHECK_EQ(cursor_, ProofCursor::kInstanceEvals);
    cursor_ = ProofCursor::kAdviceEvals;
  }

  void ReadAdviceEvals() {
    CHECK_EQ(cursor_, ProofCursor::kAdviceEvals);
    size_t num_advice_queries =
        verifying_key_.constraint_system().advice_queries().size();
    proof_.advice_evals_vec =
        base::CreateVector(num_circuits_, [this, num_advice_queries]() {
          return ReadMany<F>(num_advice_queries);
        });
    cursor_ = ProofCursor::kFixedEvals;
  }

  void ReadFixedEvals() {
    CHECK_EQ(cursor_, ProofCursor::kFixedEvals);
    size_t num_fixed_queries =
        verifying_key_.constraint_system().fixed_queries().size();
    proof_.fixed_evals = ReadMany<F>(num_fixed_queries);
    cursor_ = ProofCursor::kVanishingEval;
  }

  void ReadVanishingEval() {
    CHECK_EQ(cursor_, ProofCursor::kVanishingEval);
    proof_.vanishing_eval = Read<F>();
    cursor_ = ProofCursor::kCommonPermutationEvals;
  }

  void ReadCommonPermutationEvals() {
    CHECK_EQ(cursor_, ProofCursor::kCommonPermutationEvals);
    proof_.common_permutation_evals = ReadMany<F>(
        verifying_key_.permutation_verifying_key().commitments().size());
    cursor_ = ProofCursor::kPermutationEvals;
  }

  void ReadPermutationEvals() {
    CHECK_EQ(cursor_, ProofCursor::kPermutationEvals);
    proof_.permutation_product_evals_vec.resize(num_circuits_);
    proof_.permutation_product_next_evals_vec.resize(num_circuits_);
    proof_.permutation_product_last_evals_vec.resize(num_circuits_);
    for (size_t i = 0; i < num_circuits_; ++i) {
      size_t size = proof_.permutation_product_commitments_vec[i].size();
      proof_.permutation_product_evals_vec[i].reserve(size);
      proof_.permutation_product_next_evals_vec[i].reserve(size);
      proof_.permutation_product_last_evals_vec[i].reserve(size);
      for (size_t j = 0; j < size; ++j) {
        proof_.permutation_product_evals_vec[i].push_back(Read<F>());
        proof_.permutation_product_next_evals_vec[i].push_back(Read<F>());
        proof_.permutation_product_last_evals_vec[i].push_back(
            (j != size - 1) ? std::optional<F>(Read<F>()) : std::optional<F>());
      }
    }
    cursor_ = ProofCursor::kLookupEvalsVec;
  }

  void ReadLookupEvals() {
    CHECK_EQ(cursor_, ProofCursor::kLookupEvalsVec);
    proof_.lookup_product_evals_vec.resize(num_circuits_);
    proof_.lookup_product_next_evals_vec.resize(num_circuits_);
    proof_.lookup_permuted_input_evals_vec.resize(num_circuits_);
    proof_.lookup_permuted_input_inv_evals_vec.resize(num_circuits_);
    proof_.lookup_permuted_table_evals_vec.resize(num_circuits_);
    for (size_t i = 0; i < num_circuits_; ++i) {
      size_t size = proof_.lookup_product_commitments_vec[i].size();
      proof_.lookup_product_evals_vec[i].reserve(size);
      proof_.lookup_product_next_evals_vec[i].reserve(size);
      proof_.lookup_permuted_input_evals_vec[i].reserve(size);
      proof_.lookup_permuted_input_inv_evals_vec[i].reserve(size);
      proof_.lookup_permuted_table_evals_vec[i].reserve(size);
      for (size_t j = 0; j < size; ++j) {
        proof_.lookup_product_evals_vec[i].push_back(Read<F>());
        proof_.lookup_product_next_evals_vec[i].push_back(Read<F>());
        proof_.lookup_permuted_input_evals_vec[i].push_back(Read<F>());
        proof_.lookup_permuted_input_inv_evals_vec[i].push_back(Read<F>());
        proof_.lookup_permuted_table_evals_vec[i].push_back(Read<F>());
      }
    }
    // TODO(chokobole): Implement reading data for the last pairing step.
  }

 private:
  template <typename T>
  T Read() {
    T value;
    CHECK(transcript_->ReadFromProof(&value));
    return value;
  }

  template <typename T>
  std::vector<T> ReadMany(size_t n) {
    return base::CreateVector(n, [this]() { return Read<T>(); });
  }

  const VerifyingKey<PCS>& verifying_key_;
  // not owned
  crypto::TranscriptReader<C>* const transcript_ = nullptr;
  size_t num_circuits_ = 0;
  Proof<F, C> proof_;
  ProofCursor cursor_ = ProofCursor::kAdviceCommitmentsVecAndChallenges;
};

}  // namespace tachyon::zk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_PROOF_READER_H_
