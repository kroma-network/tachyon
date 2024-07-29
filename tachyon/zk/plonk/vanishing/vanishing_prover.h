// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_VANISHING_VANISHING_PROVER_H_
#define TACHYON_ZK_PLONK_VANISHING_VANISHING_PROVER_H_

#include <vector>

#include "absl/types/span.h"

#include "tachyon/crypto/commitments/polynomial_openings.h"
#include "tachyon/zk/base/blinded_polynomial.h"
#include "tachyon/zk/base/entities/prover_base.h"
#include "tachyon/zk/plonk/base/multi_phase_ref_table.h"
#include "tachyon/zk/plonk/keys/proving_key.h"
#include "tachyon/zk/plonk/permutation/permutation_prover.h"
#include "tachyon/zk/shuffle/prover.h"

namespace tachyon::zk::plonk {

template <typename Poly, typename Evals, typename ExtendedPoly,
          typename ExtendedEvals>
class VanishingProver {
 public:
  using F = typename Poly::Field;

  template <typename PCS>
  void CreateRandomPoly(ProverBase<PCS>* prover);

  constexpr static size_t GetNumRandomPolyCommitment() { return 1; }

  template <typename PCS>
  void CommitRandomPoly(ProverBase<PCS>* prover, size_t& commit_idx) const;

  template <typename PCS, typename LS,
            typename LookupProver = typename LS::Prover>
  void CreateHEvals(
      ProverBase<PCS>* prover, const ProvingKey<LS>& proving_key,
      const std::vector<MultiPhaseRefTable<Poly>>& tables, const F& theta,
      const F& beta, const F& gamma, const F& y,
      const std::vector<PermutationProver<Poly, Evals>>& permutation_provers,
      const std::vector<LookupProver>& lookup_provers,
      const std::vector<shuffle::Prover<Poly, Evals>>& shuffle_provers);

  template <typename PCS>
  void CreateFinalHPoly(ProverBase<PCS>* prover,
                        const ConstraintSystem<F>& constraint_system);

  template <typename F>
  static size_t GetNumFinalHPolyCommitment(
      const ConstraintSystem<F>& constraint_system) {
    return constraint_system.ComputeDegree() - 1;
  }

  template <typename PCS>
  void CommitFinalHPoly(ProverBase<PCS>* prover,
                        const ConstraintSystem<F>& constraint_system,
                        size_t& commit_idx);

  template <typename PCS>
  void BatchEvaluate(ProverBase<PCS>* prover,
                     const ConstraintSystem<F>& constraint_system,
                     const std::vector<MultiPhaseRefTable<Poly>>& tables,
                     const F& x, const F& x_n);

  template <typename PCS, typename Domain>
  static void OpenAdviceInstanceColumns(
      const Domain* domain, const ConstraintSystem<F>& constraint_system,
      const MultiPhaseRefTable<Poly>& tables, const F& x,
      std::vector<crypto::PolynomialOpening<Poly>>& openings);

  template <typename Domain>
  static void OpenFixedColumns(
      const Domain* domain, const ConstraintSystem<F>& constraint_system,
      const MultiPhaseRefTable<Poly>& tables, const F& x,
      std::vector<crypto::PolynomialOpening<Poly>>& openings);

  void Open(const F& x,
            std::vector<crypto::PolynomialOpening<Poly>>& openings) const;

 private:
  template <typename PCS, ColumnType C>
  static void EvaluateColumns(ProverBase<PCS>* prover,
                              const absl::Span<const Poly> polys,
                              const std::vector<QueryData<C>>& queries,
                              const F& x);

  template <typename Domain, ColumnType C>
  static void OpenColumns(
      const Domain* domain, const absl::Span<const Poly> polys,
      const std::vector<QueryData<C>>& queries, const F& x,
      std::vector<crypto::PolynomialOpening<Poly>>& openings);

  BlindedPolynomial<Poly, Evals> random_poly_;
  ExtendedEvals h_evals_;
  ExtendedPoly h_poly_;
  Poly combined_h_poly_;
  std::vector<F> h_blinds_;
};

}  // namespace tachyon::zk::plonk

#include "tachyon/zk/plonk/vanishing/vanishing_prover_impl.h"

#endif  // TACHYON_ZK_PLONK_VANISHING_VANISHING_PROVER_H_
