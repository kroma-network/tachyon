// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_PROVER_H_
#define TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_PROVER_H_

#include <stddef.h>

#include <functional>
#include <vector>

#include "tachyon/base/logging.h"
#include "tachyon/crypto/commitments/polynomial_openings.h"
#include "tachyon/zk/base/blinded_polynomial.h"
#include "tachyon/zk/base/entities/prover_base.h"
#include "tachyon/zk/plonk/base/ref_table.h"
#include "tachyon/zk/plonk/permutation/permutation_argument.h"
#include "tachyon/zk/plonk/permutation/permutation_opening_point_set.h"
#include "tachyon/zk/plonk/permutation/permutation_proving_key.h"
#include "tachyon/zk/plonk/permutation/permutation_table_store.h"

namespace tachyon::zk::plonk {

template <typename Poly, typename Evals>
class PermutationProver {
 public:
  using F = typename Poly::Field;

  const std::vector<BlindedPolynomial<Poly, Evals>>& grand_product_polys()
      const {
    return grand_product_polys_;
  }

  // Creates Zₚ,ᵢ for chunk index i.
  //
  // See Halo2 book to figure out logic in detail.
  // https://zcash.github.io/halo2/design/proving-system/permutation.html
  template <typename PCS>
  static void BatchCreateGrandProductPolys(
      std::vector<PermutationProver>& permutation_provers,
      ProverBase<PCS>* prover, const PermutationArgument& argument,
      const std::vector<RefTable<Evals>>& tables,
      size_t constraint_system_degree,
      const PermutationProvingKey<Poly, Evals>& permutation_proving_key,
      const F& beta, const F& gamma);

  constexpr static size_t GetNumGrandProductPolysCommitments(
      const std::vector<PermutationProver>& permutation_provers) {
    if (permutation_provers.empty()) return 0;
    return permutation_provers.size() *
           permutation_provers[0].grand_product_polys_.size();
  }

  template <typename PCS>
  static void BatchCommitGrandProductPolys(
      const std::vector<PermutationProver>& permutation_provers,
      ProverBase<PCS>* prover, size_t& commit_idx);

  template <typename Domain>
  static void TransformEvalsToPoly(
      std::vector<PermutationProver>& permutation_provers,
      const Domain* domain) {
    VLOG(2) << "Transform permutation virtual columns to polys";
    for (PermutationProver& permutation_prover : permutation_provers) {
      permutation_prover.TransformEvalsToPoly(domain);
    }
  }

  template <typename PCS>
  static void BatchEvaluate(
      const std::vector<PermutationProver>& permutation_provers,
      ProverBase<PCS>* prover, const PermutationOpeningPointSet<F>& point_set) {
    for (const PermutationProver& permutation_prover : permutation_provers) {
      permutation_prover.Evaluate(prover, point_set);
    }
  }

  template <typename PCS>
  static void EvaluateProvingKey(
      ProverBase<PCS>* prover,
      const PermutationProvingKey<Poly, Evals>& proving_key,
      const PermutationOpeningPointSet<F>& point_set);

  constexpr static size_t GetNumOpenings(
      const std::vector<PermutationProver>& permutation_provers,
      const PermutationProvingKey<Poly, Evals>& proving_key) {
    if (permutation_provers.empty()) return 0;
    if (permutation_provers[0].grand_product_polys_.empty()) return 0;
    return permutation_provers.size() *
               (permutation_provers[0].grand_product_polys_.size() * 3 - 1) +
           proving_key.permutations().size();
  }

  void Open(const PermutationOpeningPointSet<F>& point_set,
            std::vector<crypto::PolynomialOpening<Poly>>& openings) const;

  static void OpenPermutationProvingKey(
      const PermutationProvingKey<Poly, Evals>& proving_key,
      const PermutationOpeningPointSet<F>& point_set,
      std::vector<crypto::PolynomialOpening<Poly>>& openings);

 private:
  template <typename PCS>
  void CreateGrandProductPolys(ProverBase<PCS>* prover,
                               const PermutationTableStore<Evals>& table_store,
                               size_t chunk_num, const F& beta, const F& gamma);

  template <typename Domain>
  void TransformEvalsToPoly(const Domain* domain);

  template <typename PCS>
  void Evaluate(ProverBase<PCS>* prover,
                const PermutationOpeningPointSet<F>& point_set) const;

  static std::function<F(size_t, RowIndex)> CreateNumeratorCallback(
      const std::vector<base::Ref<const Evals>>& unpermuted_columns,
      const std::vector<base::Ref<const Evals>>& value_columns, const F& beta,
      const F& gamma);

  static std::function<F(size_t, RowIndex)> CreateDenominatorCallback(
      const std::vector<base::Ref<const Evals>>& permuted_columns,
      const std::vector<base::Ref<const Evals>>& value_columns, const F& beta,
      const F& gamma);

  std::vector<BlindedPolynomial<Poly, Evals>> grand_product_polys_;
};

}  // namespace tachyon::zk::plonk

#include "tachyon/zk/plonk/permutation/permutation_prover_impl.h"

#endif  // TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_PROVER_H_
