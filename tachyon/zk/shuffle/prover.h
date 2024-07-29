// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_SHUFFLE_PROVER_H_
#define TACHYON_ZK_SHUFFLE_PROVER_H_

#include <stddef.h>

#include <vector>

#include "absl/types/span.h"

#include "tachyon/crypto/commitments/polynomial_openings.h"
#include "tachyon/zk/base/blinded_polynomial.h"
#include "tachyon/zk/base/entities/prover_base.h"
#include "tachyon/zk/plonk/base/multi_phase_ref_table.h"
#include "tachyon/zk/plonk/expressions/proving_evaluator.h"
#include "tachyon/zk/shuffle/argument.h"
#include "tachyon/zk/shuffle/opening_point_set.h"
#include "tachyon/zk/shuffle/pair.h"

namespace tachyon::zk::shuffle {

template <typename Poly, typename Evals>
class Prover {
 public:
  using F = typename Poly::Field;

  const std::vector<Pair<Evals>>& compressed_pairs() const {
    return compressed_pairs_;
  }
  const std::vector<BlindedPolynomial<Poly, Evals>>& grand_product_polys()
      const {
    return grand_product_polys_;
  }

  template <typename Domain>
  static void BatchCompressPairs(
      std::vector<Prover>& shuffle_provers, const Domain* domain,
      const std::vector<Argument<F>>& arguments, const F& theta,
      const std::vector<plonk::MultiPhaseRefTable<Evals>>& tables);

  template <typename PCS>
  static void BatchPermutePairs(std::vector<Prover>& shuffle_provers,
                                ProverBase<PCS>* prover) {
    for (Prover& shuffle_prover : shuffle_provers) {
      shuffle_prover.PermutePairs(prover);
    }
  }

  template <typename PCS>
  static void BatchCreateGrandProductPolys(std::vector<Prover>& shuffle_provers,
                                           ProverBase<PCS>* prover,
                                           const F& gamma) {
    for (Prover& shuffle_prover : shuffle_provers) {
      shuffle_prover.CreateGrandProductPolys(prover, gamma);
    }
  }

  constexpr static size_t GetNumGrandProductPolysCommitments(
      const std::vector<Prover>& shuffle_provers) {
    if (shuffle_provers.empty()) return 0;
    return shuffle_provers.size() *
           shuffle_provers[0].grand_product_polys_.size();
  }

  template <typename PCS>
  static void BatchCommitGrandProductPolys(
      const std::vector<Prover>& shuffle_provers, ProverBase<PCS>* prover,
      size_t& commit_idx);

  template <typename Domain>
  static void TransformEvalsToPoly(std::vector<Prover>& shuffle_provers,
                                   const Domain* domain) {
    VLOG(2) << "Transform shuffle virtual columns to polys";
    for (Prover& shuffle_prover : shuffle_provers) {
      shuffle_prover.TransformEvalsToPoly(domain);
    }
  }

  template <typename PCS>
  static void BatchEvaluate(const std::vector<Prover>& shuffle_provers,
                            ProverBase<PCS>* prover,
                            const OpeningPointSet<F>& point_set) {
    for (const Prover& shuffle_prover : shuffle_provers) {
      shuffle_prover.Evaluate(prover, point_set);
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
  static BlindedPolynomial<Poly, Evals> CreateGrandProductPoly(
      ProverBase<PCS>* prover, const Pair<Evals>& compressed_pair,
      const F& gamma);

  template <typename Domain>
  void CompressPairs(const Domain* domain,
                     const std::vector<Argument<F>>& arguments, const F& theta,
                     const plonk::ProvingEvaluator<Evals>& evaluator_tpl);

  template <typename PCS>
  void CreateGrandProductPolys(ProverBase<PCS>* prover, const F& gamma);

  template <typename Domain>
  void TransformEvalsToPoly(const Domain* domain);

  template <typename PCS>
  void Evaluate(ProverBase<PCS>* prover,
                const OpeningPointSet<F>& point_set) const;

  static std::function<F(RowIndex)> CreateNumeratorCallback(const Evals& input,
                                                            const F& gamma);

  static std::function<F(RowIndex)> CreateDenominatorCallback(
      const Evals& input, const F& gamma);

  // A_compressedᵢ(X), S_compressedᵢ(X)
  std::vector<Pair<Evals>> compressed_pairs_;
  // Zₛ,ᵢ(X)
  std::vector<BlindedPolynomial<Poly, Evals>> grand_product_polys_;
};

}  // namespace tachyon::zk::shuffle

#include "tachyon/zk/shuffle/prover_impl.h"

#endif  // TACHYON_ZK_SHUFFLE_PROVER_H_
