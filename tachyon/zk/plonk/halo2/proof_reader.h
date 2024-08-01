#ifndef TACHYON_ZK_PLONK_HALO2_PROOF_READER_H_
#define TACHYON_ZK_PLONK_HALO2_PROOF_READER_H_

#include <utility>
#include <vector>

#include "tachyon/base/logging.h"
#include "tachyon/crypto/transcripts/transcript.h"
#include "tachyon/zk/plonk/halo2/proof.h"
#include "tachyon/zk/plonk/keys/verifying_key.h"
#include "tachyon/zk/plonk/permutation/permutation_utils.h"

namespace tachyon::zk::plonk::halo2 {

enum class ProofCursor {
  kAdviceCommitmentsVecAndChallenges,
  kTheta,
  kLookupPreparedCommitments,
  kBetaAndGamma,
  kPermutationProductCommitments,
  kLookupGrandCommitments,
  kShuffleGrandCommitments,
  kVanishingRandomPolyCommitment,
  kY,
  kVanishingHPolyCommitments,
  kX,
  kInstanceEvals,
  kAdviceEvals,
  kFixedEvals,
  kVanishingRandomEval,
  kCommonPermutationEvals,
  kPermutationEvals,
  kLookupEvalsVec,
  kShuffleEvalsVec,
  kDone,
};

template <lookup::Type Type, typename F, typename C>
class ProofReader {
 public:
  using Proof = halo2::Proof<Type, F, C>;

  ProofReader(const VerifyingKey<F, C>& verifying_key,
              crypto::TranscriptReader<C>* transcript, size_t num_circuits)
      : verifying_key_(verifying_key),
        transcript_(transcript),
        num_circuits_(num_circuits) {}

  const Proof& proof() const { return proof_; }
  Proof& proof() { return proof_; }

  void ReadAdviceCommitmentsVecAndChallenges() {
    CHECK_EQ(cursor_, ProofCursor::kAdviceCommitmentsVecAndChallenges);
    const ConstraintSystem<F>& constraint_system =
        verifying_key_.constraint_system();
    proof_.advices_commitments_vec.resize(num_circuits_);
    size_t num_advice_columns = constraint_system.num_advice_columns();
    for (size_t i = 0; i < num_circuits_; ++i) {
      proof_.advices_commitments_vec[i].resize(num_advice_columns);
    }
    proof_.challenges.reserve(constraint_system.challenge_phases().size());
    for (Phase current_phase : constraint_system.GetPhases()) {
      for (size_t i = 0; i < num_circuits_; ++i) {
        const std::vector<Phase>& advice_column_phases =
            constraint_system.advice_column_phases();
        for (size_t j = 0; j < num_advice_columns; ++j) {
          if (current_phase == advice_column_phases[j]) {
            proof_.advices_commitments_vec[i][j] = Read<C>();
          }
        }
      }
      for (Phase phase : constraint_system.challenge_phases()) {
        if (current_phase == phase) {
          F challenge = transcript_->SqueezeChallenge();
          VLOG(2) << "Halo2(challenge[" << phase.value()
                  << "]): " << challenge.ToHexString(true);
          proof_.challenges.push_back(std::move(challenge));
        }
      }
    }
    cursor_ = ProofCursor::kTheta;
  }

  void ReadTheta() {
    CHECK_EQ(cursor_, ProofCursor::kTheta);
    proof_.theta = transcript_->SqueezeChallenge();
    VLOG(2) << "Halo2(theta): " << proof_.theta.ToHexString(true);
    cursor_ = ProofCursor::kLookupPreparedCommitments;
  }

  void ReadLookupPreparedCommitments() {
    CHECK_EQ(cursor_, ProofCursor::kLookupPreparedCommitments);
    size_t num_lookups = verifying_key_.constraint_system().lookups().size();
    if constexpr (Type == lookup::Type::kHalo2) {
      proof_.lookup_permuted_commitments_vec =
          base::CreateVector(num_circuits_, [this, num_lookups]() {
            return base::CreateVector(num_lookups, [this]() {
              C input = Read<C>();
              C table = Read<C>();
              return lookup::Pair<C>(std::move(input), std::move(table));
            });
          });
    } else if constexpr (Type == lookup::Type::kLogDerivativeHalo2) {
      proof_.lookup_m_poly_commitments_vec = base::CreateVector(
          num_circuits_,
          [this, num_lookups]() { return ReadMany<C>(num_lookups); });
    } else {
      NOTREACHED();
    }
    cursor_ = ProofCursor::kBetaAndGamma;
  }

  void ReadBetaAndGamma() {
    CHECK_EQ(cursor_, ProofCursor::kBetaAndGamma);
    proof_.beta = transcript_->SqueezeChallenge();
    VLOG(2) << "Halo2(beta): " << proof_.beta.ToHexString(true);
    proof_.gamma = transcript_->SqueezeChallenge();
    VLOG(2) << "Halo2(gamma): " << proof_.gamma.ToHexString(true);
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
    cursor_ = ProofCursor::kLookupGrandCommitments;
  }

  void ReadLookupGrandCommitments() {
    CHECK_EQ(cursor_, ProofCursor::kLookupGrandCommitments);
    size_t num_lookups = verifying_key_.constraint_system().lookups().size();
    if constexpr (Type == lookup::Type::kHalo2) {
      proof_.lookup_product_commitments_vec = base::CreateVector(
          num_circuits_,
          [this, num_lookups]() { return ReadMany<C>(num_lookups); });
    } else if constexpr (Type == lookup::Type::kLogDerivativeHalo2) {
      proof_.lookup_sum_commitments_vec = base::CreateVector(
          num_circuits_,
          [this, num_lookups]() { return ReadMany<C>(num_lookups); });
    } else {
      NOTREACHED();
    }

    cursor_ = ProofCursor::kShuffleGrandCommitments;
  }

  void ReadShuffleGrandCommitments() {
    CHECK_EQ(cursor_, ProofCursor::kShuffleGrandCommitments);
    size_t num_shuffles = verifying_key_.constraint_system().shuffles().size();
    proof_.shuffle_product_commitments_vec = base::CreateVector(
        num_circuits_,
        [this, num_shuffles]() { return ReadMany<C>(num_shuffles); });
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
    VLOG(2) << "Halo2(y): " << proof_.y.ToHexString(true);
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
    VLOG(2) << "Halo2(x): " << proof_.x.ToHexString(true);
    cursor_ = ProofCursor::kInstanceEvals;
  }

  void ReadInstanceEvalsIfQueryInstance() {
    CHECK_EQ(cursor_, ProofCursor::kInstanceEvals);
    size_t num_instance_queries =
        verifying_key_.constraint_system().instance_queries().size();
    proof_.instance_evals_vec =
        base::CreateVector(num_circuits_, [this, num_instance_queries]() {
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
    cursor_ = ProofCursor::kVanishingRandomEval;
  }

  void ReadVanishingRandomEval() {
    CHECK_EQ(cursor_, ProofCursor::kVanishingRandomEval);
    proof_.vanishing_random_eval = Read<F>();
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

    if constexpr (Type == lookup::Type::kHalo2) {
      proof_.lookup_product_evals_vec.resize(num_circuits_);
      proof_.lookup_product_next_evals_vec.resize(num_circuits_);
      proof_.lookup_permuted_input_evals_vec.resize(num_circuits_);
      proof_.lookup_permuted_input_prev_evals_vec.resize(num_circuits_);
      proof_.lookup_permuted_table_evals_vec.resize(num_circuits_);
      for (size_t i = 0; i < num_circuits_; ++i) {
        size_t size = proof_.lookup_product_commitments_vec[i].size();
        proof_.lookup_product_evals_vec[i].reserve(size);
        proof_.lookup_product_next_evals_vec[i].reserve(size);
        proof_.lookup_permuted_input_evals_vec[i].reserve(size);
        proof_.lookup_permuted_input_prev_evals_vec[i].reserve(size);
        proof_.lookup_permuted_table_evals_vec[i].reserve(size);
        for (size_t j = 0; j < size; ++j) {
          proof_.lookup_product_evals_vec[i].push_back(Read<F>());
          proof_.lookup_product_next_evals_vec[i].push_back(Read<F>());
          proof_.lookup_permuted_input_evals_vec[i].push_back(Read<F>());
          proof_.lookup_permuted_input_prev_evals_vec[i].push_back(Read<F>());
          proof_.lookup_permuted_table_evals_vec[i].push_back(Read<F>());
        }
      }
    } else if constexpr (Type == lookup::Type::kLogDerivativeHalo2) {
      proof_.lookup_sum_evals_vec.resize(num_circuits_);
      proof_.lookup_sum_next_evals_vec.resize(num_circuits_);
      proof_.lookup_m_evals_vec.resize(num_circuits_);
      for (size_t i = 0; i < num_circuits_; ++i) {
        size_t size = proof_.lookup_sum_commitments_vec[i].size();
        proof_.lookup_sum_evals_vec.reserve(size);
        proof_.lookup_sum_next_evals_vec.reserve(size);
        proof_.lookup_m_evals_vec.reserve(size);
        for (size_t j = 0; j < size; ++j) {
          proof_.lookup_sum_evals_vec[i].push_back(Read<F>());
          proof_.lookup_sum_next_evals_vec[i].push_back(Read<F>());
          proof_.lookup_m_evals_vec[i].push_back(Read<F>());
        }
      }
    } else {
      NOTREACHED();
    }

    cursor_ = ProofCursor::kShuffleEvalsVec;
  }

  void ReadShuffleEvals() {
    CHECK_EQ(cursor_, ProofCursor::kShuffleEvalsVec);

    proof_.shuffle_product_evals_vec.resize(num_circuits_);
    proof_.shuffle_product_next_evals_vec.resize(num_circuits_);
    for (size_t i = 0; i < num_circuits_; ++i) {
      size_t size = proof_.shuffle_product_commitments_vec[i].size();
      proof_.shuffle_product_evals_vec[i].reserve(size);
      proof_.shuffle_product_next_evals_vec[i].reserve(size);
      for (size_t j = 0; j < size; ++j) {
        proof_.shuffle_product_evals_vec[i].push_back(Read<F>());
        proof_.shuffle_product_next_evals_vec[i].push_back(Read<F>());
      }
    }

    cursor_ = ProofCursor::kDone;
  }

  bool Done() const { return cursor_ == ProofCursor::kDone; }

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

  const VerifyingKey<F, C>& verifying_key_;
  // not owned
  crypto::TranscriptReader<C>* const transcript_ = nullptr;
  size_t num_circuits_ = 0;
  Proof proof_;
  ProofCursor cursor_ = ProofCursor::kAdviceCommitmentsVecAndChallenges;
};

}  // namespace tachyon::zk::plonk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_PROOF_READER_H_
