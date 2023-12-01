// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_BASE_PROVER_H_
#define TACHYON_ZK_BASE_PROVER_H_

#include <memory>
#include <utility>

#include "tachyon/zk/base/blinded_polynomial.h"
#include "tachyon/zk/base/blinder.h"
#include "tachyon/zk/base/entity.h"

namespace tachyon::zk {

template <typename PCSTy, typename ExtendedDomain>
class Prover : public Entity<PCSTy, ExtendedDomain> {
 public:
  using F = typename PCSTy::Field;
  using Domain = typename PCSTy::Domain;
  using Evals = typename PCSTy::Evals;
  using Poly = typename PCSTy::Poly;
  using Commitment = typename PCSTy::Commitment;

  Prover(PCSTy pcs, std::unique_ptr<Domain> domain,
         std::unique_ptr<ExtendedDomain> extended_domain,
         std::unique_ptr<TranscriptWriter<Commitment>> writer,
         Blinder<PCSTy> blinder)
      : Entity<PCSTy, ExtendedDomain>(std::move(pcs), std::move(domain),
                                      std::move(extended_domain),
                                      std::move(writer)),
        blinder_(std::move(blinder)) {}

  Blinder<PCSTy>& blinder() { return blinder_; }

  TranscriptWriter<Commitment>* GetWriter() {
    return static_cast<TranscriptWriter<Commitment>*>(this->transcript());
  }

  bool CommitEvalsWithBlind(const Evals& evals, BlindedPolynomial<Poly>* out) {
    if (evals.NumElements() != this->domain_->size()) return false;

    Commitment commitment;
    if (!this->pcs_.CommitLagrange(evals, &commitment)) return false;
    GetWriter()->WriteToProof(commitment);

    *out = {this->domain_->IFFT(evals), blinder_.Generate()};
    return true;
  }

  void Evaluate(const Poly& poly, const F& x) {
    F result = poly.Evaluate(x);
    GetWriter()->WriteToProof(result);
  }

 protected:
  Blinder<PCSTy> blinder_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_PROVER_H_
