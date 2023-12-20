// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_HALO2_PROVER_H_
#define TACHYON_ZK_PLONK_HALO2_PROVER_H_

#include <memory>
#include <utility>

#include "tachyon/zk/base/entities/prover_base.h"
#include "tachyon/zk/plonk/halo2/random_field_generator.h"
#include "tachyon/zk/plonk/halo2/verifier.h"

namespace tachyon::zk::halo2 {

template <typename PCSTy>
class Prover : public ProverBase<PCSTy> {
 public:
  using F = typename PCSTy::Field;
  using Evals = typename PCSTy::Evals;
  using Commitment = typename PCSTy::Commitment;

  static Prover CreateFromRandomSeed(
      PCSTy&& pcs, std::unique_ptr<crypto::TranscriptWriter<Commitment>> writer,
      size_t blinding_factors) {
    auto rng = std::make_unique<crypto::XORShiftRNG>(
        crypto::XORShiftRNG::FromRandomSeed());
    return CreateFromRNG(std::move(pcs), std::move(writer), std::move(rng),
                         blinding_factors);
  }

  static Prover CreateFromSeed(
      PCSTy&& pcs, std::unique_ptr<crypto::TranscriptWriter<Commitment>> writer,
      const uint8_t seed[16], size_t blinding_factors) {
    auto rng = std::make_unique<crypto::XORShiftRNG>(
        crypto::XORShiftRNG::FromSeed(seed));
    return CreateFromRNG(std::move(pcs), std::move(writer), std::move(rng),
                         blinding_factors);
  }

  static Prover CreateFromRNG(
      PCSTy&& pcs, std::unique_ptr<crypto::TranscriptWriter<Commitment>> writer,
      std::unique_ptr<crypto::XORShiftRNG> rng, size_t blinding_factors) {
    auto generator = std::make_unique<RandomFieldGenerator<F>>(rng.get());
    Blinder<PCSTy> blinder(generator.get(), blinding_factors);
    return {std::move(pcs), std::move(writer), std::move(blinder),
            std::move(rng), std::move(generator)};
  }

  crypto::XORShiftRNG* rng() { return rng_.get(); }
  RandomFieldGenerator<F>* generator() { return generator_.get(); }

  std::unique_ptr<Verifier<PCSTy>> ToVerifier(
      std::unique_ptr<crypto::TranscriptReader<Commitment>> reader) {
    std::unique_ptr<Verifier<PCSTy>> ret = std::make_unique<Verifier<PCSTy>>(
        std::move(this->pcs_), std::move(reader));
    ret->set_domain(std::move(this->domain_));
    ret->set_extended_domain(std::move(this->extended_domain_));
    return ret;
  }

 private:
  Prover(PCSTy&& pcs,
         std::unique_ptr<crypto::TranscriptWriter<Commitment>> writer,
         Blinder<PCSTy>&& blinder, std::unique_ptr<crypto::XORShiftRNG> rng,
         std::unique_ptr<RandomFieldGenerator<F>> generator)
      : ProverBase<PCSTy>(std::move(pcs), std::move(writer),
                          std::move(blinder)),
        rng_(std::move(rng)),
        generator_(std::move(generator)) {}

  std::unique_ptr<crypto::XORShiftRNG> rng_;
  std::unique_ptr<RandomFieldGenerator<F>> generator_;
};

}  // namespace tachyon::zk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_PROVER_H_
