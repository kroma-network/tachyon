// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_BASE_ENTITIES_ENTITY_H_
#define TACHYON_ZK_BASE_ENTITIES_ENTITY_H_

#include <memory>
#include <utility>

#include "tachyon/crypto/transcripts/transcript.h"

namespace tachyon::zk {

// |Entity| class is a parent class of |Prover| and |Verifier|.
//
// - If you write codes only for prover, you should use |Prover| class.
// - If you write codes only for verifier, you should use |Verifier| class.
// - If you write codes for both prover and verifier, you should use
//  |Entity| class.
template <typename _PCSTy>
class Entity {
 public:
  using PCSTy = _PCSTy;
  using F = typename PCSTy::Field;
  using Domain = typename PCSTy::Domain;
  using ExtendedDomain = typename PCSTy::ExtendedDomain;
  using Evals = typename PCSTy::Evals;
  using Poly = typename PCSTy::Poly;
  using Commitment = typename PCSTy::Commitment;

  Entity(PCSTy&& pcs, std::unique_ptr<Domain> domain,
         std::unique_ptr<ExtendedDomain> extended_domain,
         std::unique_ptr<crypto::Transcript<Commitment>> transcript)
      : pcs_(std::move(pcs)),
        domain_(std::move(domain)),
        extended_domain_(std::move(extended_domain)),
        transcript_(std::move(transcript)) {}

  const PCSTy& pcs() const { return pcs_; }
  PCSTy& pcs() { return pcs_; }
  const Domain* domain() const { return domain_.get(); }
  const ExtendedDomain* extended_domain() const {
    return extended_domain_.get();
  }
  crypto::Transcript<Commitment>* transcript() { return transcript_.get(); }

 protected:
  PCSTy pcs_;
  std::unique_ptr<Domain> domain_;
  std::unique_ptr<ExtendedDomain> extended_domain_;
  std::unique_ptr<crypto::Transcript<Commitment>> transcript_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_ENTITIES_ENTITY_H_
