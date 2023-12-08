// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_BASE_ENTITIES_VERIFIER_H_
#define TACHYON_ZK_BASE_ENTITIES_VERIFIER_H_

#include <memory>
#include <utility>

#include "tachyon/zk/base/entities/entity.h"

namespace tachyon::zk {

template <typename PCSTy>
class Verifier : public Entity<PCSTy> {
 public:
  using Domain = typename PCSTy::Domain;
  using ExtendedDomain = typename PCSTy::ExtendedDomain;
  using Commitment = typename PCSTy::Commitment;

  Verifier(PCSTy&& pcs, std::unique_ptr<Domain> domain,
           std::unique_ptr<ExtendedDomain> extended_domain,
           std::unique_ptr<crypto::TranscriptReader<Commitment>> transcript)
      : Entity<PCSTy>(std::move(pcs), std::move(domain),
                      std::move(extended_domain), std::move(transcript)) {}

  crypto::TranscriptReader<Commitment>* GetReader() {
    return this->transcript()->ToReader();
  }
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_ENTITIES_VERIFIER_H_
