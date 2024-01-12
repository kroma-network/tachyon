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

template <typename F, size_t MaxDegree, typename TranscriptReader,
          typename TranscriptWriter>
class FRI final : public UnivariatePolynomialCommitmentScheme<
                      FRI<F, MaxDegree, TranscriptReader, TranscriptWriter>> {
 public:
  using Base = UnivariatePolynomialCommitmentScheme<
      FRI<F, MaxDegree, TranscriptReader, TranscriptWriter>>;
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

  [[nodiscard]] bool Commit(const Poly& poly, std::vector<F>* roots,
                            TranscriptWriter* writer) const {
    size_t num_layers = domain_->log_size_of_group();
    BinaryMerkleTree<F, F, MaxDegree + 1> tree(storage_->GetLayer(0), hasher_);
    Evals evals = domain_->FFT(poly);
    roots->resize(num_layers + 1);
    if (!tree.Commit(evals.evaluations(), &(*roots)[0])) return false;
    if (!writer->template WriteToProof</*NeedToWriteToTranscript=*/true>(
            (*roots)[0]))
      return false;
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
        if (!tree.Commit(evals.evaluations(), &(*roots)[i])) return false;
        if (!writer->template WriteToProof</*NeedToWriteToTranscript=*/true>(
                (*roots)[i]))
          return false;
        cur_poly = &folded_poly;
      }
    }

    beta = writer->SqueezeChallenge();
    folded_poly = cur_poly->template Fold<false>(beta);
    const F* constant = folded_poly[0];
    (*roots)[num_layers] = constant ? *constant : F::Zero();
    return writer->template WriteToProof</*NeedToWriteToTranscript=*/true>(
        (*roots)[num_layers]);
  }

  [[nodiscard]] bool DoCreateOpeningProof(size_t index,
                                          FRIProof<F>* fri_proof) const {
    size_t domain_size = domain_->size();
    size_t num_layers = domain_->log_size_of_group();
    fri_proof->proof.resize(num_layers);
    fri_proof->proof_sym.resize(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
      // Merkle proof for Pᵢ(ωʲ) against Cᵢ
      size_t leaf_index = index % domain_size;
      BinaryMerkleTreeStorage<F, F>* layer = storage_->GetLayer(i);
      BinaryMerkleTree<F, F, MaxDegree + 1> tree(layer, hasher_);
      if (!tree.CreateOpeningProof(leaf_index, &fri_proof->proof[i]))
        return false;

      // Merkle proof for Pᵢ(-ωʲ) against Cᵢ
      size_t half_domain_size = domain_size >> 1;
      size_t leaf_index_sym = (index + half_domain_size) % domain_size;
      if (!tree.CreateOpeningProof(leaf_index_sym, &fri_proof->proof_sym[i]))
        return false;

      domain_size = half_domain_size;
    }
    return true;
  }

  [[nodiscard]] bool DoVerifyOpeningProof(size_t index,
                                          const FRIProof<F>& fri_proof,
                                          TranscriptReader& reader) const {
    size_t domain_size = domain_->size();
    size_t num_layers = domain_->log_size_of_group();
    F root;
    F x;
    F beta;
    F evaluation;
    F evaluation_sym;
    F two_inv = F(2).Inverse();
    for (size_t i = 0; i < num_layers; ++i) {
      BinaryMerkleTreeStorage<F, F>* layer = storage_->GetLayer(i);
      BinaryMerkleTree<F, F, MaxDegree + 1> tree(layer, hasher_);

      if (!reader.template ReadFromProof</*NeedToWriteToTranscript=*/true>(
              &root))
        return false;
      if (!tree.VerifyOpeningProof(root, fri_proof.proof[i])) return false;
      if (!tree.VerifyOpeningProof(root, fri_proof.proof_sym[i])) return false;

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
        evaluation = fri_proof.proof[i].value;
        evaluation_sym = fri_proof.proof_sym[i].value;
        x = domain_->GetElement(leaf_index);
      } else {
        evaluation *= (F::One() + beta);
        evaluation_sym *= (F::One() - beta);
        evaluation += evaluation_sym;
        evaluation *= two_inv;

        if (evaluation != fri_proof.proof[i].value) {
          LOG(ERROR)
              << "Proof doesn't match with expected evaluation at layer [" << i
              << "]";
          return false;
        }
        evaluation_sym = fri_proof.proof_sym[i].value;
        x = sub_domains_[i - 1]->GetElement(leaf_index);
      }
      beta = reader.SqueezeChallenge();
      beta *= x.Inverse();
      domain_size = domain_size >> 1;
    }

    evaluation *= (F::One() + beta);
    evaluation_sym *= (F::One() - beta);
    evaluation += evaluation_sym;
    evaluation *= two_inv;

    if (!reader.template ReadFromProof</*NeedToWriteToTranscript=*/true>(&root))
      return false;
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
  std::vector<std::unique_ptr<Domain>> sub_domains_;
};

template <typename F, size_t MaxDegree, typename _TranscriptReader,
          typename _TranscriptWriter>
struct VectorCommitmentSchemeTraits<
    FRI<F, MaxDegree, _TranscriptReader, _TranscriptWriter>> {
 public:
  constexpr static size_t kMaxSize = MaxDegree + 1;
  constexpr static bool kIsTransparent = true;
  constexpr static bool kIsCommitInteractive = true;
  constexpr static bool kIsOpenInteractive = false;

  using Field = F;
  using Commitment = std::vector<F>;
  using TranscriptReader = _TranscriptReader;
  using TranscriptWriter = _TranscriptWriter;
  using Proof = FRIProof<F>;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_FRI_FRI_H_
