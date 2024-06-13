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

#include "tachyon/base/types/always_false.h"
#include "tachyon/zk/base/entities/prover_base.h"
#include "tachyon/zk/plonk/halo2/argument_data.h"
#include "tachyon/zk/plonk/halo2/c_prover_impl_base_forward.h"
#include "tachyon/zk/plonk/halo2/random_field_generator.h"
#include "tachyon/zk/plonk/halo2/verifier.h"
#include "tachyon/zk/plonk/permutation/permutation_prover.h"
#include "tachyon/zk/plonk/vanishing/vanishing_prover.h"

namespace tachyon::zk::plonk::halo2 {

template <typename PCS, typename LS>
class Prover : public ProverBase<PCS> {
 public:
  using F = typename PCS::Field;
  using Poly = typename PCS::Poly;
  using Evals = typename PCS::Evals;
  using Domain = typename PCS::Domain;
  using ExtendedPoly = typename PCS::ExtendedPoly;
  using ExtendedEvals = typename PCS::ExtendedEvals;
  using Commitment = typename PCS::Commitment;
  using LookupProver = typename LS::Prover;

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

  Verifier<PCS, LS> ToVerifier(
      std::unique_ptr<crypto::TranscriptReader<Commitment>> reader) {
    Verifier<PCS, LS> ret(std::move(this->pcs_), std::move(reader));
    ret.set_domain(std::move(this->domain_));
    ret.set_extended_domain(std::move(this->extended_domain_));
    return ret;
  }

  template <typename Circuit>
  void CreateProof(ProvingKey<LS>& proving_key,
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
  friend class c::zk::plonk::halo2::ProverImplBase<PCS, LS>;

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

  void CreateProof(ProvingKey<LS>& proving_key,
                   ArgumentData<Poly, Evals>* argument_data) {
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

    size_t num_circuits = argument_data->GetNumCircuits();
    std::vector<LookupProver> lookup_provers(num_circuits);
    std::vector<PermutationProver<Poly, Evals>> permutation_provers(
        num_circuits);
    VanishingProver<Poly, Evals, ExtendedPoly, ExtendedEvals> vanishing_prover;
    const ConstraintSystem<F>& cs =
        proving_key.verifying_key().constraint_system();
    const Domain* domain = this->domain();

    crypto::TranscriptWriter<Commitment>* writer = this->GetWriter();
    F theta = writer->SqueezeChallenge();
    VLOG(2) << "Halo2(theta): " << theta.ToHexString(true);

    std::vector<MultiPhaseRefTable<Evals>> column_tables =
        argument_data->ExportColumnTables(proving_key.fixed_columns());

    size_t commit_idx = 0;
    if constexpr (LS::type == lookup::Type::kHalo2) {
      LookupProver::BatchCompressPairs(lookup_provers, domain, cs.lookups(),
                                       theta, column_tables);
      LookupProver::BatchPermutePairs(lookup_provers, this);

      if constexpr (PCS::kSupportsBatchMode) {
        this->pcs_.SetBatchMode(
            LookupProver::GetNumPermutedPairsCommitments(lookup_provers));
      }
      LookupProver::BatchCommitPermutedPairs(lookup_provers, this, commit_idx);
    } else if constexpr (LS::type == lookup::Type::kLogDerivativeHalo2) {
      LookupProver::BatchCompressPairs(lookup_provers, domain, cs.lookups(),
                                       theta, column_tables);
      LookupProver::BatchComputeMPolys(lookup_provers, this);

      if constexpr (PCS::kSupportsBatchMode) {
        this->pcs_.SetBatchMode(
            LookupProver::GetNumMPolysCommitments(lookup_provers));
      }
      LookupProver::BatchCommitMPolys(lookup_provers, this, commit_idx);
    } else {
      static_assert(base::AlwaysFalse<LS>);
    }

    if constexpr (PCS::kSupportsBatchMode) {
      this->RetrieveAndWriteBatchCommitmentsToProof();
    }

    F beta = writer->SqueezeChallenge();
    VLOG(2) << "Halo2(beta): " << beta.ToHexString(true);
    F gamma = writer->SqueezeChallenge();
    VLOG(2) << "Halo2(gamma): " << gamma.ToHexString(true);

    PermutationProver<Poly, Evals>::BatchCreateGrandProductPolys(
        permutation_provers, this, cs.permutation(), column_tables,
        cs.ComputeDegree(), proving_key.permutation_proving_key(), beta, gamma);

    if constexpr (LS::type == lookup::Type::kHalo2) {
      LookupProver::BatchCreateGrandProductPolys(lookup_provers, this, beta,
                                                 gamma);
    } else if constexpr (LS::type == lookup::Type::kLogDerivativeHalo2) {
      LookupProver::BatchCreateGrandSumPolys(lookup_provers, this, beta);
    } else {
      static_assert(base::AlwaysFalse<LS>);
    }

    vanishing_prover.CreateRandomPoly(this);

    if constexpr (PCS::kSupportsBatchMode) {
      size_t num_lookup_poly;
      if constexpr (LS::type == lookup::Type::kHalo2) {
        num_lookup_poly =
            LookupProver::GetNumGrandProductPolysCommitments(lookup_provers);
      } else if constexpr (LS::type == lookup::Type::kLogDerivativeHalo2) {
        num_lookup_poly =
            LookupProver::GetNumGrandSumPolysCommitments(lookup_provers);
      } else {
        static_assert(base::AlwaysFalse<LS>);
      }
      this->pcs_.SetBatchMode(
          PermutationProver<Poly, Evals>::GetNumGrandProductPolysCommitments(
              permutation_provers) +
          num_lookup_poly +
          VanishingProver<Poly, Evals, ExtendedPoly,
                          ExtendedEvals>::GetNumRandomPolyCommitment());
    }
    commit_idx = 0;
    PermutationProver<Poly, Evals>::BatchCommitGrandProductPolys(
        permutation_provers, this, commit_idx);
    if constexpr (LS::type == lookup::Type::kHalo2) {
      LookupProver::BatchCommitGrandProductPolys(lookup_provers, this,
                                                 commit_idx);
    } else if constexpr (LS::type == lookup::Type::kLogDerivativeHalo2) {
      LookupProver::BatchCommitGrandSumPolys(lookup_provers, this, commit_idx);
    } else {
      static_assert(base::AlwaysFalse<LS>);
    }
    vanishing_prover.CommitRandomPoly(this, commit_idx);
    if constexpr (PCS::kSupportsBatchMode) {
      this->RetrieveAndWriteBatchCommitmentsToProof();
    }

    F y = writer->SqueezeChallenge();
    VLOG(2) << "Halo2(y): " << y.ToHexString(true);

    argument_data->TransformEvalsToPoly(domain);
    PermutationProver<Poly, Evals>::TransformEvalsToPoly(permutation_provers,
                                                         domain);
    LookupProver::TransformEvalsToPoly(lookup_provers, domain);

    argument_data->DeallocateAllColumnsVec();
    proving_key.fixed_columns().clear();
    column_tables.clear();

    std::vector<MultiPhaseRefTable<Poly>> poly_tables =
        argument_data->ExportPolyTables(proving_key.fixed_polys());

    vanishing_prover.CreateHEvals(this, proving_key, poly_tables, theta, beta,
                                  gamma, y, permutation_provers,
                                  lookup_provers);
    vanishing_prover.CreateFinalHPoly(this, cs);

    if constexpr (PCS::kSupportsBatchMode) {
      this->pcs_.SetBatchMode(
          VanishingProver<Poly, Evals, ExtendedPoly,
                          ExtendedEvals>::GetNumFinalHPolyCommitment(cs));
    }
    commit_idx = 0;
    vanishing_prover.CommitFinalHPoly(this, cs, commit_idx);
    if constexpr (PCS::kSupportsBatchMode) {
      this->RetrieveAndWriteBatchCommitmentsToProof();
    }

    F x = writer->SqueezeChallenge();
    VLOG(2) << "Halo2(x): " << x.ToHexString(true);
    F x_prev = Rotation::Prev().RotateOmega(domain, x);
    F x_next = Rotation::Next().RotateOmega(domain, x);
    Rotation last_rotation = Rotation(this->GetLastRow());
    F x_last = last_rotation.RotateOmega(domain, x);

    PermutationOpeningPointSet<F> permutation_opening_point_set(x, x_next,
                                                                x_last);
    lookup::halo2::OpeningPointSet<F> lookup_opening_point_set(x, x_prev,
                                                               x_next);
    Evaluate(proving_key, poly_tables, vanishing_prover, permutation_provers,
             lookup_provers, permutation_opening_point_set,
             lookup_opening_point_set);

    std::vector<crypto::PolynomialOpening<Poly>> openings =
        Open(proving_key, poly_tables, vanishing_prover, permutation_provers,
             lookup_provers, permutation_opening_point_set,
             lookup_opening_point_set);
    CHECK(this->pcs_.CreateOpeningProof(openings, this->GetWriter()));
  }

  void Evaluate(
      const ProvingKey<LS>& proving_key,
      const std::vector<MultiPhaseRefTable<Poly>>& poly_tables,
      VanishingProver<Poly, Evals, ExtendedPoly, ExtendedEvals>&
          vanishing_prover,
      const std::vector<PermutationProver<Poly, Evals>>& permutation_provers,
      const std::vector<LookupProver>& lookup_provers,
      const PermutationOpeningPointSet<F>& permutation_opening_point_set,
      const lookup::halo2::OpeningPointSet<F>& lookup_opening_point_set) {
    const ConstraintSystem<F>& constraint_system =
        proving_key.verifying_key().constraint_system();

    const F& x = permutation_opening_point_set.x;
    F x_n = x.Pow(this->pcs_.N());
    vanishing_prover.BatchEvaluate(this, constraint_system, poly_tables, x,
                                   x_n);
    PermutationProver<Poly, Evals>::EvaluateProvingKey(
        this, proving_key.permutation_proving_key(),
        permutation_opening_point_set);
    PermutationProver<Poly, Evals>::BatchEvaluate(
        permutation_provers, this, permutation_opening_point_set);
    LookupProver::BatchEvaluate(lookup_provers, this, lookup_opening_point_set);
  }

  std::vector<crypto::PolynomialOpening<Poly>> Open(
      const ProvingKey<LS>& proving_key,
      const std::vector<MultiPhaseRefTable<Poly>>& poly_tables,
      const VanishingProver<Poly, Evals, ExtendedPoly, ExtendedEvals>&
          vanishing_prover,
      const std::vector<PermutationProver<Poly, Evals>>& permutation_provers,
      const std::vector<LookupProver>& lookup_provers,
      const PermutationOpeningPointSet<F>& permutation_opening_point_set,
      const lookup::halo2::OpeningPointSet<F>& lookup_opening_point_set) const {
    const ConstraintSystem<F>& constraint_system =
        proving_key.verifying_key().constraint_system();
    const Domain* domain = this->domain();

    std::vector<crypto::PolynomialOpening<Poly>> openings;
    size_t num_circuits = poly_tables.size();
    size_t size =
        GetNumVanishingOpenings<PCS>(
            num_circuits, constraint_system.advice_queries().size(),
            constraint_system.instance_queries().size(),
            constraint_system.fixed_queries().size()) +
        GetNumPermutationOpenings(
            num_circuits, permutation_provers[0].grand_product_polys().size(),
            proving_key.permutation_proving_key().permutations().size()) +
        lookup::halo2::GetNumOpenings(LS::type, lookup_provers.size(),
                                      constraint_system.lookups().size());
    openings.reserve(size);

    const F& x = permutation_opening_point_set.x;
    for (size_t i = 0; i < num_circuits; ++i) {
      VanishingProver<Poly, Evals, ExtendedPoly, ExtendedEvals>::
          template OpenAdviceInstanceColumns<PCS>(domain, constraint_system,
                                                  poly_tables[i], x, openings);
      permutation_provers[i].Open(permutation_opening_point_set, openings);
      lookup_provers[i].Open(lookup_opening_point_set, openings);
    }
    VanishingProver<Poly, Evals, ExtendedPoly, ExtendedEvals>::OpenFixedColumns(
        domain, constraint_system, poly_tables[0], x, openings);
    PermutationProver<Poly, Evals>::OpenPermutationProvingKey(
        proving_key.permutation_proving_key(), permutation_opening_point_set,
        openings);
    vanishing_prover.Open(x, openings);
    CHECK_EQ(openings.size(), size);
    return openings;
  }

  std::unique_ptr<crypto::XORShiftRNG> rng_;
  std::unique_ptr<RandomFieldGenerator<F>> generator_;
};

}  // namespace tachyon::zk::plonk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_PROVER_H_
