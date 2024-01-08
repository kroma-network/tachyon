// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_HALO2_PROVER_H_
#define TACHYON_ZK_PLONK_HALO2_PROVER_H_

#include <memory>
#include <utility>

#include "absl/memory/memory.h"

#include "tachyon/zk/base/entities/prover_base.h"
#include "tachyon/zk/plonk/halo2/random_field_generator.h"
#include "tachyon/zk/plonk/halo2/verifier.h"

namespace tachyon::zk::halo2 {

template <typename PCS>
class Prover : public ProverBase<PCS> {
 public:
  using F = typename PCS::Field;
  using Evals = typename PCS::Evals;
  using Commitment = typename PCS::Commitment;
  using TranscriptReader = typename PCS::TranscriptReader;
  using TranscriptWriter = typename PCS::TranscriptWriter;

  static std::unique_ptr<Prover> CreateFromRandomSeed(
      PCS&& pcs, std::unique_ptr<TranscriptWriter> writer,
      size_t blinding_factors) {
    auto rng = std::make_unique<crypto::XORShiftRNG>(
        crypto::XORShiftRNG::FromRandomSeed());
    return CreateFromRNG(std::move(pcs), std::move(writer), std::move(rng),
                         blinding_factors);
  }

  static std::unique_ptr<Prover> CreateFromSeed(
      PCS&& pcs, std::unique_ptr<TranscriptWriter> writer,
      const uint8_t seed[16], size_t blinding_factors) {
    auto rng = std::make_unique<crypto::XORShiftRNG>(
        crypto::XORShiftRNG::FromSeed(seed));
    return CreateFromRNG(std::move(pcs), std::move(writer), std::move(rng),
                         blinding_factors);
  }

  static std::unique_ptr<Prover> CreateFromRNG(
      PCS&& pcs, std::unique_ptr<TranscriptWriter> writer,
      std::unique_ptr<crypto::XORShiftRNG> rng, size_t blinding_factors) {
    auto generator = std::make_unique<RandomFieldGenerator<F>>(rng.get());
    Blinder<PCS> blinder(generator.get(), blinding_factors);
    return absl::WrapUnique(new Prover(std::move(pcs), std::move(writer),
                                       std::move(blinder), std::move(rng),
                                       std::move(generator)));
  }

  crypto::XORShiftRNG* rng() { return rng_.get(); }
  RandomFieldGenerator<F>* generator() { return generator_.get(); }

  std::unique_ptr<Verifier<PCS>> ToVerifier(
      std::unique_ptr<TranscriptReader> reader) {
    std::unique_ptr<Verifier<PCS>> ret = std::make_unique<Verifier<PCS>>(
        std::move(this->pcs_), std::move(reader));
    ret->set_domain(std::move(this->domain_));
    ret->set_extended_domain(std::move(this->extended_domain_));
    return ret;
  }

 private:
  Prover(PCS&& pcs, std::unique_ptr<TranscriptWriter> writer,
         Blinder<PCS>&& blinder, std::unique_ptr<crypto::XORShiftRNG> rng,
         std::unique_ptr<RandomFieldGenerator<F>> generator)
      : ProverBase<PCS>(std::move(pcs), std::move(writer), std::move(blinder)),
        rng_(std::move(rng)),
        generator_(std::move(generator)) {}

  std::unique_ptr<crypto::XORShiftRNG> rng_;
  std::unique_ptr<RandomFieldGenerator<F>> generator_;
};

}  // namespace tachyon::zk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_PROVER_H_
