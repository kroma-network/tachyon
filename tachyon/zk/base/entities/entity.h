// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_BASE_ENTITIES_ENTITY_H_
#define TACHYON_ZK_BASE_ENTITIES_ENTITY_H_

#include <limits>
#include <memory>
#include <utility>

#include "tachyon/base/logging.h"
#include "tachyon/crypto/commitments/vector_commitment_scheme_traits_forward.h"
#include "tachyon/crypto/transcripts/transcript.h"
#include "tachyon/zk/base/row_types.h"

#if TACHYON_CUDA
#include "tachyon/math/polynomials/univariate/icicle/icicle_ntt_holder.h"
#endif

namespace tachyon::zk {

// |Entity| class is a parent class of |Prover| and |Verifier|.
//
// - If you write codes only for prover, you should use |Prover| class.
// - If you write codes only for verifier, you should use |Verifier| class.
// - If you write codes for both prover and verifier, you should use
//  |Entity| class.
template <typename _PCS>
class Entity {
 public:
  using PCS = _PCS;
  using F = typename PCS::Field;
  using Domain = typename PCS::Domain;
  using ExtendedDomain = typename PCS::ExtendedDomain;
  using Evals = typename PCS::Evals;
  using Poly = typename PCS::Poly;
  using Commitment = typename PCS::Commitment;

  Entity(PCS&& pcs, std::unique_ptr<crypto::Transcript<Commitment>> transcript)
      : pcs_(std::move(pcs)), transcript_(std::move(transcript)) {}

  const PCS& pcs() const { return pcs_; }
  PCS& pcs() { return pcs_; }
  void set_domain(std::unique_ptr<Domain> domain) {
    CHECK_LE(domain->size(), size_t{std::numeric_limits<RowIndex>::max()});
    domain_ = std::move(domain);
#if TACHYON_CUDA
    if (icicle_ntt_holder_) {
      domain_->set_icicle(&icicle_ntt_holder_);
    }
#endif
  }
  const Domain* domain() const { return domain_.get(); }
  void set_extended_domain(std::unique_ptr<ExtendedDomain> extended_domain) {
    extended_domain_ = std::move(extended_domain);
#if TACHYON_CUDA
    if (icicle_ntt_holder_) {
      extended_domain_->set_icicle(&icicle_ntt_holder_);
    }
#endif
  }
  const ExtendedDomain* extended_domain() const {
    return extended_domain_.get();
  }
  crypto::Transcript<Commitment>* transcript() { return transcript_.get(); }
  const crypto::Transcript<Commitment>* transcript() const {
    return transcript_.get();
  }

#if TACHYON_CUDA
  math::IcicleNTTHolder<F>&& TakeIcicleNTTHolder() && {
    return std::move(icicle_ntt_holder_);
  }

  void set_icicle_ntt_holder(math::IcicleNTTHolder<F>&& icicle_ntt_holder) {
    CHECK(!icicle_ntt_holder_);
    icicle_ntt_holder_ = std::move(icicle_ntt_holder);
  }

  void EnableIcicleNTT() {
    if (icicle_ntt_holder_) {
      LOG(WARNING)
          << "EnableIcicleNTT() is called more than once. If you see "
             "this log while running unittests, this is intended. The first "
             "call is made in 'tachyon/zk/plonk/halo2/prover_test.h'.";
    } else {
      icicle_ntt_holder_ = math::IcicleNTTHolder<F>::Create();
      CHECK(icicle_ntt_holder_->Init(extended_domain_->group_gen()));
      domain_->set_icicle(&icicle_ntt_holder_);
      extended_domain_->set_icicle(&icicle_ntt_holder_);
    }
  }
#endif

  RowIndex GetUsableRows(RowIndex blinding_factors) const {
    return domain_->size() - (blinding_factors + 1);
  }

  constexpr static RowOffset GetLastRow(RowIndex blinding_factors) {
    return -(blinding_factors + 1);
  }

  Commitment Commit(const Poly& poly) {
    Commitment commitment;
    CHECK(pcs_.Commit(poly, &commitment));
    return commitment;
  }

  Commitment Commit(const Evals& evals) {
    Commitment commitment;
    CHECK(pcs_.CommitLagrange(evals, &commitment));
    return commitment;
  }

  template <typename Container>
  Commitment Commit(const Container& container) {
    Commitment commitment;
    CHECK(pcs_.DoCommit(container, &commitment));
    return commitment;
  }

  template <typename T = PCS,
            std::enable_if_t<crypto::VectorCommitmentSchemeTraits<
                T>::kSupportsBatchMode>* = nullptr>
  void BatchCommitAt(const Poly& poly, size_t index) {
    CHECK(pcs_.Commit(poly, index));
  }

  template <typename T = PCS,
            std::enable_if_t<crypto::VectorCommitmentSchemeTraits<
                T>::kSupportsBatchMode>* = nullptr>
  void BatchCommitAt(const Evals& evals, size_t index) {
    CHECK(pcs_.CommitLagrange(evals, index));
  }

  template <typename T = PCS, typename Container,
            std::enable_if_t<crypto::VectorCommitmentSchemeTraits<
                T>::kSupportsBatchMode>* = nullptr>
  void BatchCommitAt(const Container& container, size_t index) {
    CHECK(pcs_.DoCommit(container, pcs_.batch_commitment_state(), index));
  }

 protected:
  PCS pcs_;
  std::unique_ptr<Domain> domain_;
  std::unique_ptr<ExtendedDomain> extended_domain_;
  std::unique_ptr<crypto::Transcript<Commitment>> transcript_;
#if TACHYON_CUDA
  math::IcicleNTTHolder<F> icicle_ntt_holder_;
#endif
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_ENTITIES_ENTITY_H_
