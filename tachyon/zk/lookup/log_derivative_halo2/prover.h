// Copyright (c) 2022-2024 Scroll
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.scroll and the LICENCE-APACHE.scroll
// file.

#ifndef TACHYON_ZK_LOOKUP_LOG_DERIVATIVE_HALO2_PROVER_H_
#define TACHYON_ZK_LOOKUP_LOG_DERIVATIVE_HALO2_PROVER_H_

#include <vector>

#include "tachyon/crypto/commitments/polynomial_openings.h"
#include "tachyon/zk/base/blinded_polynomial.h"
#include "tachyon/zk/base/entities/prover_base.h"
#include "tachyon/zk/lookup/halo2/opening_point_set.h"
#include "tachyon/zk/lookup/lookup_argument.h"
#include "tachyon/zk/lookup/proving_evaluator.h"
#include "tachyon/zk/plonk/base/multi_phase_ref_table.h"

namespace tachyon::zk::lookup::log_derivative_halo2 {

template <typename Poly, typename Evals>
class Prover {
 public:
  using F = typename Poly::Field;

  const std::vector<std::vector<Evals>>& compressed_inputs_vec() const {
    return compressed_inputs_vec_;
  }
  const std::vector<Evals>& compressed_tables() const {
    return compressed_tables_;
  }
  const std::vector<BlindedPolynomial<Poly, Evals>>& m_polys() const {
    return m_polys_;
  }
  const std::vector<BlindedPolynomial<Poly, Evals>>& grand_sum_polys() const {
    return grand_sum_polys_;
  }

  template <typename Domain>
  static void BatchCompressPairs(
      std::vector<Prover>& lookup_provers, const Domain* domain,
      const std::vector<Argument<F>>& arguments, const F& theta,
      const std::vector<plonk::MultiPhaseRefTable<Evals>>& tables);

  template <typename PCS>
  static void BatchComputeMPolys(std::vector<Prover>& lookup_provers,
                                 ProverBase<PCS>* prover) {
    for (Prover& lookup_prover : lookup_provers) {
      lookup_prover.ComputeMPolys(prover);
    }
  }

  constexpr static size_t GetNumMPolysCommitments(
      const std::vector<Prover>& lookup_provers) {
    if (lookup_provers.empty()) return 0;
    return lookup_provers.size() * lookup_provers[0].m_polys_.size();
  }

  template <typename PCS>
  static void BatchCommitMPolys(const std::vector<Prover>& lookup_provers,
                                ProverBase<PCS>* prover, size_t& commit_idx);

  template <typename PCS>
  static void BatchCreateGrandSumPolys(std::vector<Prover>& lookup_provers,
                                       ProverBase<PCS>* prover, const F& beta) {
    for (Prover& lookup_prover : lookup_provers) {
      lookup_prover.CreateGrandSumPolys(prover, beta);
    }
  }

  constexpr static size_t GetNumGrandSumPolysCommitments(
      const std::vector<Prover>& lookup_provers) {
    if (lookup_provers.empty()) return 0;
    return lookup_provers.size() * lookup_provers[0].grand_sum_polys_.size();
  }

  template <typename PCS>
  static void BatchCommitGrandSumPolys(
      const std::vector<Prover>& lookup_provers, ProverBase<PCS>* prover,
      size_t& commit_idx);

  template <typename Domain>
  static void TransformEvalsToPoly(std::vector<Prover>& lookup_provers,
                                   const Domain* domain) {
    VLOG(2) << "Transform lookup virtual columns to polys";
    for (Prover& lookup_prover : lookup_provers) {
      lookup_prover.TransformEvalsToPoly(domain);
    }
  }

  template <typename PCS>
  static void BatchEvaluate(const std::vector<Prover>& lookup_provers,
                            ProverBase<PCS>* prover,
                            const halo2::OpeningPointSet<F>& point_set) {
    for (const Prover& lookup_prover : lookup_provers) {
      lookup_prover.Evaluate(prover, point_set);
    }
  }

  void Open(const halo2::OpeningPointSet<F>& point_set,
            std::vector<crypto::PolynomialOpening<Poly>>& openings) const;

 private:
  template <typename Domain>
  static std::vector<Evals> CompressInputs(
      const Domain* domain, const Argument<F>& argument, const F& theta,
      const ProvingEvaluator<Evals>& evaluator_tpl);

  template <typename Domain>
  static Evals CompressTable(const Domain* domain, const Argument<F>& argument,
                             const F& theta,
                             const ProvingEvaluator<Evals>& evaluator_tpl);

  template <typename Domain>
  void CompressPairs(const Domain* domain,
                     const std::vector<Argument<F>>& arguments, const F& theta,
                     const ProvingEvaluator<Evals>& evaluator_tpl);

  template <typename PCS>
  static BlindedPolynomial<Poly, Evals> ComputeMPoly(
      ProverBase<PCS>* prover, const std::vector<Evals>& compressed_inputs,
      const Evals& compressed_table);

  template <typename PCS>
  void ComputeMPolys(ProverBase<PCS>* prover);

  static void ComputeLogDerivatives(const Evals& evals, const F& beta,
                                    std::vector<F>& ret);

  template <typename PCS>
  static BlindedPolynomial<Poly, Evals> CreateGrandSumPoly(
      ProverBase<PCS>* prover, const Evals& m_values,
      const std::vector<Evals>& compressed_inputs,
      const Evals& compressed_table, const F& beta);

  template <typename PCS>
  void CreateGrandSumPolys(ProverBase<PCS>* prover, const F& beta);

  template <typename Domain>
  void TransformEvalsToPoly(const Domain* domain);

  template <typename PCS>
  void Evaluate(ProverBase<PCS>* prover,
                const halo2::OpeningPointSet<F>& point_set) const;

  // fᵢ(X)
  std::vector<std::vector<Evals>> compressed_inputs_vec_;
  // t(X)
  std::vector<Evals> compressed_tables_;
  // m(X)
  std::vector<BlindedPolynomial<Poly, Evals>> m_polys_;
  // ϕ(X)
  std::vector<BlindedPolynomial<Poly, Evals>> grand_sum_polys_;
};

}  // namespace tachyon::zk::lookup::log_derivative_halo2

#include "tachyon/zk/lookup/log_derivative_halo2/prover_impl.h"

#endif  // TACHYON_ZK_LOOKUP_LOG_DERIVATIVE_HALO2_PROVER_H_
