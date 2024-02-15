// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_HALO2_PROVER_H_
#define TACHYON_ZK_PLONK_HALO2_PROVER_H_

#include <memory>
#include <utility>
#include <vector>

#include "tachyon/zk/base/entities/prover_base.h"
#include "tachyon/zk/plonk/halo2/argument.h"
#include "tachyon/zk/plonk/halo2/c_prover_impl_base_forward.h"
#include "tachyon/zk/plonk/halo2/random_field_generator.h"
#include "tachyon/zk/plonk/halo2/verifier.h"

namespace tachyon::zk::plonk::halo2 {

template <typename PCS>
class Prover : public ProverBase<PCS> {
 public:
  using F = typename PCS::Field;
  using Poly = typename PCS::Poly;
  using Evals = typename PCS::Evals;
  using ExtendedEvals = typename PCS::ExtendedEvals;
  using Commitment = typename PCS::Commitment;

  static Prover CreateFromRandomSeed(
      PCS&& pcs, std::unique_ptr<crypto::TranscriptWriter<Commitment>> writer,
      RowIndex blinding_factors) {
    auto rng = std::make_unique<crypto::XORShiftRNG>(
        crypto::XORShiftRNG::FromRandomSeed());
    return CreateFromRNG(std::move(pcs), std::move(writer), std::move(rng),
                         blinding_factors);
  }

  template <typename Container>
  static Prover CreateFromSeed(
      PCS&& pcs, std::unique_ptr<crypto::TranscriptWriter<Commitment>> writer,
      const Container& seed, RowIndex blinding_factors) {
    auto rng = std::make_unique<crypto::XORShiftRNG>(
        crypto::XORShiftRNG::FromSeed(seed));
    return CreateFromRNG(std::move(pcs), std::move(writer), std::move(rng),
                         blinding_factors);
  }

  static Prover CreateFromRNG(
      PCS&& pcs, std::unique_ptr<crypto::TranscriptWriter<Commitment>> writer,
      std::unique_ptr<crypto::XORShiftRNG> rng, RowIndex blinding_factors) {
    auto generator = std::make_unique<RandomFieldGenerator<F>>(rng.get());
    Blinder<F> blinder(generator.get(), blinding_factors);
    return {std::move(pcs), std::move(writer), std::move(blinder),
            std::move(rng), std::move(generator)};
  }

  crypto::XORShiftRNG* rng() { return rng_.get(); }
  RandomFieldGenerator<F>* generator() { return generator_.get(); }

  Verifier<PCS> ToVerifier(
      std::unique_ptr<crypto::TranscriptReader<Commitment>> reader) {
    Verifier<PCS> ret(std::move(this->pcs_), std::move(reader));
    ret.set_domain(std::move(this->domain_));
    ret.set_extended_domain(std::move(this->extended_domain_));
    return ret;
  }

  template <typename Circuit>
  void CreateProof(ProvingKey<Poly, Evals, Commitment>& proving_key,
                   std::vector<std::vector<Evals>>&& instance_columns_vec,
                   std::vector<Circuit>& circuits) {
    size_t num_circuits = circuits.size();

    // Check length of instances.
    CHECK_EQ(num_circuits, instance_columns_vec.size());
    for (const std::vector<Evals>& instances_vec : instance_columns_vec) {
      CHECK_EQ(instances_vec.size(), proving_key.verifying_key()
                                         .constraint_system()
                                         .num_instance_columns());
    }

    // Initially write hash value of verification key to transcript.
    crypto::TranscriptWriter<Commitment>* writer = this->GetWriter();
    CHECK(writer->WriteToTranscript(
        proving_key.verifying_key().transcript_repr()));

    // It owns all the columns, polys and the others required in the proof
    // generation process and provides step-by-step logics as its methods.
    ArgumentData<Poly, Evals> argument_data = ArgumentData<Poly, Evals>::Create(
        this, circuits, proving_key.verifying_key().constraint_system(),
        std::move(instance_columns_vec));
    CreateProof(proving_key, &argument_data);
  }

 private:
  friend class c::zk::plonk::halo2::ProverImplBase<PCS>;

  Prover(PCS&& pcs,
         std::unique_ptr<crypto::TranscriptWriter<Commitment>> writer,
         Blinder<F>&& blinder, std::unique_ptr<crypto::XORShiftRNG> rng,
         std::unique_ptr<RandomFieldGenerator<F>> generator)
      : ProverBase<PCS>(std::move(pcs), std::move(writer), std::move(blinder)),
        rng_(std::move(rng)),
        generator_(std::move(generator)) {}

  void SetRng(std::unique_ptr<crypto::XORShiftRNG> rng) {
    rng_ = std::move(rng);
    generator_ = std::make_unique<RandomFieldGenerator<F>>(rng_.get());
    this->blinder_ =
        Blinder<F>(generator_.get(), this->blinder_.blinding_factors());
  }

  void CreateProof(ProvingKey<Poly, Evals, Commitment>& proving_key,
                   ArgumentData<Poly, Evals>* argument_data) {
    using ExtendedPoly = typename PCS::ExtendedPoly;

    Argument<Poly, Evals> argument(&proving_key.fixed_columns(),
                                   &proving_key.fixed_polys(), argument_data);
    // NOTE(chokobole): This is an entry point fom Halo2 rust. So this is the
    // earliest time to log constraint system.
    VLOG(1) << "PCS name: " << this->pcs_.Name() << ", k: " << this->pcs_.K()
            << ", n: " << this->pcs_.N() << ", extended_k: "
            << proving_key.verifying_key().constraint_system().ComputeExtendedK(
                   this->pcs_.K())
            << ", max_degree: " << PCS::kMaxDegree
            << ", extended_max_degree: " << PCS::kMaxExtendedDegree;
    VLOG(1) << "Halo2 Constraint System: "
            << proving_key.verifying_key().constraint_system().ToString();

    crypto::TranscriptWriter<Commitment>* writer = this->GetWriter();
    F theta = writer->SqueezeChallenge();
    VLOG(2) << "Halo2(theta): " << theta.ToHexString(true);
    std::vector<std::vector<lookup::halo2::LookupPermuted<Poly, Evals>>>
        permuted_lookups_vec = argument.PermuteLookupsStep(
            this, proving_key.verifying_key().constraint_system(), theta);

    F beta = writer->SqueezeChallenge();
    VLOG(2) << "Halo2(beta): " << beta.ToHexString(true);
    F gamma = writer->SqueezeChallenge();
    VLOG(2) << "Halo2(gamma): " << gamma.ToHexString(true);
    StepReturns<PermutationCommitted<Poly>,
                lookup::halo2::LookupCommitted<Poly>, VanishingCommitted<Poly>>
        committed_result = argument.CommitCircuitStep(
            this, proving_key.verifying_key().constraint_system(),
            proving_key.permutation_proving_key(),
            std::move(permuted_lookups_vec), beta, gamma);

    F y = writer->SqueezeChallenge();
    VLOG(2) << "Halo2(y): " << y.ToHexString(true);
    argument.TransformAdvice(this->domain());
    argument.DeallocateAllColumnsVec();
    ExtendedEvals circuit_column = argument.GenerateCircuitPolynomial(
        this, proving_key, committed_result, beta, gamma, theta, y);
    VanishingConstructed<Poly, ExtendedPoly> constructed_vanishing =
        CommitFinalHPoly(this, std::move(committed_result).TakeVanishing(),
                         proving_key.verifying_key(), circuit_column);

    F x = writer->SqueezeChallenge();
    VLOG(2) << "Halo2(x): " << x.ToHexString(true);
    StepReturns<PermutationEvaluated<Poly>,
                lookup::halo2::LookupEvaluated<Poly>, VanishingEvaluated<Poly>>
        evaluated_result =
            argument.EvaluateCircuitStep(this, proving_key, committed_result,
                                         std::move(constructed_vanishing), x);

    std::vector<crypto::PolynomialOpening<Poly>> openings =
        argument.ConstructOpenings(this, proving_key, evaluated_result, x);

    CHECK(this->pcs_.CreateOpeningProof(openings, this->GetWriter()));
  }

  std::unique_ptr<crypto::XORShiftRNG> rng_;
  std::unique_ptr<RandomFieldGenerator<F>> generator_;
};

}  // namespace tachyon::zk::plonk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_PROVER_H_
