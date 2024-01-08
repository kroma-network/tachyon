// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_BASE_ENTITIES_ENTITY_H_
#define TACHYON_ZK_BASE_ENTITIES_ENTITY_H_

#include <memory>
#include <utility>

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
  using TranscriptReader = typename PCS::TranscriptReader;
  using TranscriptWriter = typename PCS::TranscriptWriter;

  explicit Entity(PCS&& pcs) : pcs_(std::move(pcs)) {}
  virtual ~Entity() = default;

  const PCS& pcs() const { return pcs_; }
  PCS& pcs() { return pcs_; }
  void set_domain(std::unique_ptr<Domain> domain) {
    domain_ = std::move(domain);
  }
  const Domain* domain() const { return domain_.get(); }
  void set_extended_domain(std::unique_ptr<ExtendedDomain> extended_domain) {
    extended_domain_ = std::move(extended_domain);
  }
  const ExtendedDomain* extended_domain() const {
    return extended_domain_.get();
  }

  virtual TranscriptReader* GetReader() const = 0;
  virtual TranscriptWriter* GetWriter() const = 0;

 protected:
  PCS pcs_;
  std::unique_ptr<Domain> domain_;
  std::unique_ptr<ExtendedDomain> extended_domain_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_ENTITIES_ENTITY_H_
