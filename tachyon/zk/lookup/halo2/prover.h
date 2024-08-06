// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_LOOKUP_HALO2_PROVER_H_
#define TACHYON_ZK_LOOKUP_HALO2_PROVER_H_

#include <stddef.h>

#include <vector>

#include "absl/types/span.h"

#include "tachyon/base/profiler.h"
#include "tachyon/crypto/commitments/polynomial_openings.h"
#include "tachyon/zk/base/blinded_polynomial.h"
#include "tachyon/zk/base/entities/prover_base.h"
#include "tachyon/zk/lookup/argument.h"
#include "tachyon/zk/lookup/halo2/opening_point_set.h"
#include "tachyon/zk/lookup/pair.h"
#include "tachyon/zk/plonk/base/multi_phase_ref_table.h"
#include "tachyon/zk/plonk/expressions/proving_evaluator.h"

namespace tachyon::zk::lookup::halo2 {

template <typename Poly, typename Evals>
class Prover {
 public:
  using F = typename Poly::Field;

  const std::vector<Pair<Evals>>& compressed_pairs() const {
    return compressed_pairs_;
  }
  const std::vector<Pair<BlindedPolynomial<Poly, Evals>>>& permuted_pairs()
      const {
    return permuted_pairs_;
  }
  const std::vector<BlindedPolynomial<Poly, Evals>>& grand_product_polys()
      const {
    return grand_product_polys_;
  }

  template <typename Domain>
  static void BatchCompressPairs(
      std::vector<Prover>& lookup_provers, const Domain* domain,
      const std::vector<Argument<F>>& arguments, const F& theta,
      const std::vector<plonk::MultiPhaseRefTable<Evals>>& tables);

  template <typename PCS>
  static void BatchPermutePairs(std::vector<Prover>& lookup_provers,
                                ProverBase<PCS>* prover) {
    TRACE_EVENT("ProofGeneration", "Lookup::Halo2::Prover::BatchPermutePairs");
    for (Prover& lookup_prover : lookup_provers) {
      lookup_prover.PermutePairs(prover);
    }
  }

  constexpr static size_t GetNumPermutedPairsCommitments(
      const std::vector<Prover>& lookup_provers) {
    if (lookup_provers.empty()) return 0;
    return lookup_provers.size() * lookup_provers[0].permuted_pairs_.size() * 2;
  }

  template <typename PCS>
  static void BatchCommitPermutedPairs(
      const std::vector<Prover>& lookup_provers, ProverBase<PCS>* prover,
      size_t& commit_idx);

  template <typename PCS>
  static void BatchCreateGrandProductPolys(std::vector<Prover>& lookup_provers,
                                           ProverBase<PCS>* prover,
                                           const F& beta, const F& gamma) {
    TRACE_EVENT("ProofGeneration",
                "Lookup::Halo2::Prover::BatchCreateGrandProductPolys");
    for (Prover& lookup_prover : lookup_provers) {
      lookup_prover.CreateGrandProductPolys(prover, beta, gamma);
    }
  }

  constexpr static size_t GetNumGrandProductPolysCommitments(
      const std::vector<Prover>& lookup_provers) {
    if (lookup_provers.empty()) return 0;
    return lookup_provers.size() *
           lookup_provers[0].grand_product_polys_.size();
  }

  template <typename PCS>
  static void BatchCommitGrandProductPolys(
      const std::vector<Prover>& lookup_provers, ProverBase<PCS>* prover,
      size_t& commit_idx);

  template <typename Domain>
  static void TransformEvalsToPoly(std::vector<Prover>& lookup_provers,
                                   const Domain* domain) {
    TRACE_EVENT("ProofGeneration",
                "Lookup::Halo2::Prover::TransformEvalsToPoly");
    VLOG(2) << "Transform lookup virtual columns to polys";
    for (Prover& lookup_prover : lookup_provers) {
      lookup_prover.TransformEvalsToPoly(domain);
    }
  }

  template <typename PCS>
  static void BatchEvaluate(const std::vector<Prover>& lookup_provers,
                            ProverBase<PCS>* prover,
                            const OpeningPointSet<F>& point_set) {
    TRACE_EVENT("ProofGeneration", "Lookup::Halo2::Prover::BatchEvaluate");
    for (const Prover& lookup_prover : lookup_provers) {
      lookup_prover.Evaluate(prover, point_set);
    }
  }

  void Open(const OpeningPointSet<F>& point_set,
            std::vector<crypto::PolynomialOpening<Poly>>& openings) const;

 private:
  template <typename Domain>
  static Pair<Evals> CompressPair(
      const Domain* domain, const Argument<F>& argument, const F& theta,
      const plonk::ProvingEvaluator<Evals>& evaluator_tpl);

  template <typename PCS>
  static Pair<BlindedPolynomial<Poly, Evals>> PermutePair(
      ProverBase<PCS>* prover, const Pair<Evals>& compressed_pair);

  template <typename PCS>
  static BlindedPolynomial<Poly, Evals> CreateGrandProductPoly(
      ProverBase<PCS>* prover, const Pair<Evals>& compressed_pair,
      const Pair<BlindedPolynomial<Poly, Evals>>& permuted_pair, const F& beta,
      const F& gamma);

  template <typename Domain>
  void CompressPairs(const Domain* domain,
                     const std::vector<Argument<F>>& arguments, const F& theta,
                     const plonk::ProvingEvaluator<Evals>& evaluator_tpl);

  template <typename PCS>
  void PermutePairs(ProverBase<PCS>* prover);

  template <typename PCS>
  void CreateGrandProductPolys(ProverBase<PCS>* prover, const F& beta,
                               const F& gamma);

  template <typename Domain>
  void TransformEvalsToPoly(const Domain* domain);

  template <typename PCS>
  void Evaluate(ProverBase<PCS>* prover,
                const OpeningPointSet<F>& point_set) const;

  static std::function<F(RowIndex)> CreateNumeratorCallback(
      const Pair<Evals>& compressed_pair, const F& beta, const F& gamma);

  static std::function<F(RowIndex)> CreateDenominatorCallback(
      const Pair<BlindedPolynomial<Poly, Evals>>& permuted_pair, const F& beta,
      const F& gamma);

  // A_compressedᵢ(X), S_compressedᵢ(X)
  std::vector<Pair<Evals>> compressed_pairs_;
  // A'ᵢ(X), S'ᵢ(X)
  std::vector<Pair<BlindedPolynomial<Poly, Evals>>> permuted_pairs_;
  // Zₗ,ᵢ(X)
  std::vector<BlindedPolynomial<Poly, Evals>> grand_product_polys_;
};

}  // namespace tachyon::zk::lookup::halo2

#include "tachyon/zk/lookup/halo2/prover_impl.h"

#endif  // TACHYON_ZK_LOOKUP_HALO2_PROVER_H_
