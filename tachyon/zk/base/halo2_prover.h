// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_BASE_HALO2_PROVER_H_
#define TACHYON_ZK_BASE_HALO2_PROVER_H_

#include <memory>
#include <utility>

#include "tachyon/zk/base/halo2_random_field_generator.h"
#include "tachyon/zk/base/prover.h"

namespace tachyon::zk {

template <typename PCSTy, typename ExtendedDomain>
class Halo2Prover : public Prover<PCSTy, ExtendedDomain> {
 public:
  using F = typename PCSTy::Field;
  using Domain = typename PCSTy::Domain;
  using Evals = typename PCSTy::Evals;
  using Commitment = typename PCSTy::Commitment;

  static Halo2Prover CreateFromRandomSeed(
      PCSTy pcs, std::unique_ptr<Domain> domain,
      std::unique_ptr<ExtendedDomain> extended_domain,
      std::unique_ptr<TranscriptWriter<Commitment>> writer,
      size_t blinding_factors) {
    auto rng = std::make_unique<crypto::XORShiftRNG>(
        crypto::XORShiftRNG::FromRandomSeed());
    return CreateFromRNG(std::move(pcs), std::move(domain),
                         std::move(extended_domain), std::move(writer),
                         std::move(rng), blinding_factors);
  }

  static Halo2Prover CreateFromSeed(
      PCSTy pcs, std::unique_ptr<Domain> domain,
      std::unique_ptr<ExtendedDomain> extended_domain,
      std::unique_ptr<TranscriptWriter<Commitment>> writer, uint8_t seed[16],
      size_t blinding_factors) {
    auto rng = std::make_unique<crypto::XORShiftRNG>(
        crypto::XORShiftRNG::FromSeed(seed));
    return CreateFromRNG(std::move(pcs), std::move(domain),
                         std::move(extended_domain), std::move(writer),
                         std::move(rng), blinding_factors);
  }

  static Halo2Prover CreateFromRNG(
      PCSTy pcs, std::unique_ptr<Domain> domain,
      std::unique_ptr<ExtendedDomain> extended_domain,
      std::unique_ptr<TranscriptWriter<Commitment>> writer,
      std::unique_ptr<crypto::XORShiftRNG> rng, size_t blinding_factors) {
    auto generator = std::make_unique<Halo2RandomFieldGenerator<F>>(rng.get());
    Blinder<PCSTy> blinder(generator.get(), blinding_factors);
    return {std::move(pcs),      std::move(domain), std::move(extended_domain),
            std::move(blinder),  std::move(writer), std::move(rng),
            std::move(generator)};
  }

  crypto::XORShiftRNG* rng() { return rng_.get(); }
  Halo2RandomFieldGenerator<F>* generator() { return generator_.get(); }

 private:
  Halo2Prover(PCSTy pcs, std::unique_ptr<Domain> domain,
              std::unique_ptr<ExtendedDomain> extended_domain,
              Blinder<PCSTy> blinder,
              std::unique_ptr<TranscriptWriter<Commitment>> writer,
              std::unique_ptr<crypto::XORShiftRNG> rng,
              std::unique_ptr<Halo2RandomFieldGenerator<F>> generator)
      : Prover<PCSTy, ExtendedDomain>(std::move(pcs), std::move(domain),
                                      std::move(extended_domain),
                                      std::move(blinder), std::move(writer)),
        rng_(std::move(rng)),
        generator_(std::move(generator)) {}

  std::unique_ptr<crypto::XORShiftRNG> rng_;
  std::unique_ptr<Halo2RandomFieldGenerator<F>> generator_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_HALO2_PROVER_H_
