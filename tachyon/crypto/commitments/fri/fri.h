// Use of this source code is governed by a Apache-2.0 style license that
// can be found in the LICENSE.lambdaworks.

#ifndef TACHYON_CRYPTO_COMMITMENTS_FRI_FRI_H_
#define TACHYON_CRYPTO_COMMITMENTS_FRI_FRI_H_

#include <memory>
#include <utility>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/crypto/commitments/fri/fri_proof.h"
#include "tachyon/crypto/commitments/fri/fri_storage.h"
#include "tachyon/crypto/commitments/merkle_tree/binary_merkle_tree/binary_merkle_tree.h"
#include "tachyon/crypto/commitments/univariate_polynomial_commitment_scheme.h"
#include "tachyon/crypto/transcripts/transcript.h"

namespace tachyon::crypto {

template <typename F, size_t MaxDegree>
class FRI : public UnivariatePolynomialCommitmentScheme<FRI<F, MaxDegree>> {
 public:
  using Base = UnivariatePolynomialCommitmentScheme<FRI<F, MaxDegree>>;
  using Poly = typename Base::Poly;
  using Evals = typename Base::Evals;
  using Domain = typename Base::Domain;

  FRI() = default;
  FRI(const Domain* domain, FRIStorage<F>* storage,
      BinaryMerkleHasher<F, F>* hasher)
      : domain_(domain), storage_(storage), hasher_(hasher) {
    // This ensures last folding process.
    CHECK_GE(domain->size(), size_t{2}) << "Domain size must be at least 2";
    size_t k = domain->log_size_of_group();
    if (k > 1) {
      sub_domains_ = base::CreateVector(k - 1, [k](size_t i) {
        return Domain::Create(size_t{1} << (k - i - 1));
      });
    }
    storage_->Allocate(k);
  }

  // UnivariatePolynomialCommitmentScheme methods
  size_t N() const { return domain_->size(); }

  [[nodiscard]] bool Commit(const Poly& poly, Transcript<F>* transcript) const {
    size_t num_layers = domain_->log_size_of_group();
    TranscriptWriter<F>* writer = transcript->ToWriter();
    BinaryMerkleTree<F, F, MaxDegree + 1> tree(storage_->GetLayer(0), hasher_);
    Evals evals = domain_->FFT(poly);
    F root;
    if (!tree.Commit(evals.evaluations(), &root)) return false;
    if (!writer->WriteToProof(root)) return false;
    const Poly* cur_poly = &poly;

    F beta;
    Poly folded_poly;
    if (num_layers > 1) {
      for (size_t i = 1; i < num_layers; ++i) {
        // Pᵢ(X)   = Pᵢ_even(X²) + X * Pᵢ_odd(X²)
        // Pᵢ₊₁(X) = Pᵢ_even(X²) + β * Pᵢ_odd(X²)
        beta = writer->SqueezeChallenge();
        folded_poly = cur_poly->template Fold<false>(beta);
        BinaryMerkleTree<F, F, MaxDegree + 1> tree(storage_->GetLayer(i),
                                                   hasher_);
        evals = sub_domains_[i - 1]->FFT(folded_poly);
        if (!tree.Commit(evals.evaluations(), &root)) return false;
        if (!writer->WriteToProof(root)) return false;
        cur_poly = &folded_poly;
      }
    }

    beta = writer->SqueezeChallenge();
    folded_poly = cur_poly->template Fold<false>(beta);
    const F* constant = folded_poly[0];
    root = constant ? *constant : F::Zero();
    return writer->WriteToProof(root);
  }

  [[nodiscard]] bool DoCreateOpeningProof(size_t index,
                                          FRIProof<F>* fri_proof) const {
    size_t domain_size = domain_->size();
    size_t num_layers = domain_->log_size_of_group();
    for (size_t i = 0; i < num_layers; ++i) {
      size_t leaf_index = index % domain_size;
      BinaryMerkleTreeStorage<F>* layer = storage_->GetLayer(i);
      BinaryMerkleTree<F, F, MaxDegree + 1> tree(layer, hasher_);
      BinaryMerkleProof<F> proof;
      if (!tree.CreateOpeningProof(leaf_index, &proof)) return false;
      // Merkle proof for Pᵢ(ωʲ) against Cᵢ
      fri_proof->paths.push_back(std::move(proof));
      // Pᵢ(ωʲ)
      fri_proof->evaluations.push_back(
          layer->GetHash(domain_size - 1 + leaf_index));

      size_t half_domain_size = domain_size >> 1;
      size_t leaf_index_sym = (index + half_domain_size) % domain_size;
      BinaryMerkleProof<F> proof_sym;
      if (!tree.CreateOpeningProof(leaf_index_sym, &proof_sym)) return false;
      // Merkle proof for Pᵢ(-ωʲ) against Cᵢ
      fri_proof->paths_sym.push_back(std::move(proof_sym));
      // Pᵢ(-ωʲ)
      fri_proof->evaluations_sym.push_back(
          layer->GetHash(domain_size - 1 + leaf_index_sym));

      domain_size = half_domain_size;
    }
    return true;
  }

  [[nodiscard]] bool DoVerifyOpeningProof(Transcript<F>& transcript,
                                          size_t index,
                                          const FRIProof<F>& proof) const {
    TranscriptReader<F>* reader = transcript.ToReader();
    size_t domain_size = domain_->size();
    size_t num_layers = domain_->log_size_of_group();
    F root;
    F x;
    F beta;
    F evaluation;
    F evaluation_sym;
    F two_inv = F(2).Inverse();
    for (size_t i = 0; i < num_layers; ++i) {
      BinaryMerkleTreeStorage<F>* layer = storage_->GetLayer(i);
      BinaryMerkleTree<F, F, MaxDegree + 1> tree(layer, hasher_);

      if (!reader->ReadFromProof(&root)) return false;
      if (!tree.VerifyOpeningProof(root, proof.evaluations[i], proof.paths[i]))
        return false;

      if (!tree.VerifyOpeningProof(root, proof.evaluations_sym[i],
                                   proof.paths_sym[i]))
        return false;

      // Given equations:
      // Pᵢ(X)  = Pᵢ_even(X²) + X * Pᵢ_odd(X²)
      // Pᵢ(-X) = Pᵢ_even(X²) - X * Pᵢ_odd(X²)
      //
      // Using Gaussian elimination, we derive:
      // Pᵢ_even(X²) = (Pᵢ(X) + Pᵢ(-X)) / 2
      // Pᵢ_odd(X²)  = (Pᵢ(X) - Pᵢ(-X)) / (2 * X)
      //
      // Next layer equation:
      // Pᵢ₊₁(X) = Pᵢ_even(X²) + β * Pᵢ_odd(X²)
      //
      // If the domain of Pᵢ(X) is Dᵢ = {ω⁰, ω¹, ..., ωⁿ⁻¹},
      // then the domain of Pᵢ₊₁(X) is Dᵢ₊₁ = {ω⁰, ω¹, ..., ωᵏ⁻¹},
      // where k = n / 2.
      //
      // As per the definition:
      // Pᵢ₊₁(ωʲ) = Pᵢ_even((ωʲ)²) + β * Pᵢ_odd((ωʲ)²)
      //
      // Substituting Pᵢ_even and Pᵢ_odd:
      // Pᵢ₊₁(ωʲ) = (Pᵢ(ωʲ) + Pᵢ(-ωʲ)) / 2 + β * (Pᵢ(ωʲ) - Pᵢ(-ωʲ)) / (2 * ωʲ)
      //           = ((1 + β * ω⁻ʲ) * Pᵢ(ωʲ) + (1 - β * ω⁻ʲ) * Pᵢ(-ωʲ)) / 2
      size_t leaf_index = index % domain_size;
      if (i == 0) {
        evaluation = proof.evaluations[i];
        evaluation_sym = proof.evaluations_sym[i];
        x = domain_->GetElement(leaf_index);
      } else {
        evaluation *= (F::One() + beta);
        evaluation_sym *= (F::One() - beta);
        evaluation += evaluation_sym;
        evaluation *= two_inv;

        if (evaluation != proof.evaluations[i]) {
          LOG(ERROR)
              << "Proof doesn't match with expected evaluation at layer [" << i
              << "]";
          return false;
        }
        evaluation_sym = proof.evaluations_sym[i];
        x = sub_domains_[i - 1]->GetElement(leaf_index);
      }
      beta = reader->SqueezeChallenge();
      beta *= x.Inverse();
      domain_size = domain_size >> 1;
    }

    evaluation *= (F::One() + beta);
    evaluation_sym *= (F::One() - beta);
    evaluation += evaluation_sym;
    evaluation *= two_inv;

    if (!reader->ReadFromProof(&root)) return false;
    if (root != evaluation) {
      LOG(ERROR) << "Root doesn't match with expected evaluation";
      return false;
    }
    return true;
  }

 private:
  // not owned
  const Domain* domain_ = nullptr;
  // not owned
  mutable FRIStorage<F>* storage_ = nullptr;
  // not owned
  BinaryMerkleHasher<F, F>* hasher_ = nullptr;
  // not owned
  Transcript<F>* transcript_ = nullptr;
  std::vector<std::unique_ptr<Domain>> sub_domains_;
};

template <typename F, size_t MaxDegree>
struct VectorCommitmentSchemeTraits<FRI<F, MaxDegree>> {
 public:
  constexpr static size_t kMaxSize = MaxDegree + 1;
  constexpr static bool kIsTransparent = true;

  using Field = F;
  using Commitment = Transcript<F>;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_FRI_FRI_H_
