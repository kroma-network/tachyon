// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_VANISHING_VANISHING_VERIFIER_H_
#define TACHYON_ZK_PLONK_VANISHING_VANISHING_VERIFIER_H_

#include <memory>
#include <vector>

#include "tachyon/crypto/commitments/polynomial_openings.h"
#include "tachyon/zk/lookup/verifying_evaluator.h"
#include "tachyon/zk/plonk/constraint_system/constraint_system.h"
#include "tachyon/zk/plonk/vanishing/vanishing_verifier_data.h"

namespace tachyon::zk::plonk {

template <typename F, typename C>
class VanishingVerifier {
 public:
  explicit VanishingVerifier(const VanishingVerifierData<F, C>& data)
      : data_(data) {}

  void Evaluate(const ConstraintSystem<F>& constraint_system,
                std::vector<F>& evals) {
    lookup::VerifyingEvaluator<F> evaluator(data_);
    for (const Gate<F>& gate : constraint_system.gates()) {
      for (const std::unique_ptr<Expression<F>>& poly : gate.polys()) {
        evals.push_back(poly->Evaluate(&evaluator));
      }
    }
  }

  template <typename PCS, typename Poly, typename Domain>
  void OpenAdviceInstanceColumns(
      const Domain* domain, const F& x,
      const ConstraintSystem<F>& constraint_system,
      std::vector<crypto::PolynomialOpening<Poly, C>>& openings) const {
    if constexpr (PCS::kQueryInstance) {
      OpenColumns(domain, x, constraint_system.instance_queries(),
                  data_.instance_commitments, data_.instance_evals, openings);
    }
    OpenColumns(domain, x, constraint_system.advice_queries(),
                data_.advice_commitments, data_.advice_evals, openings);
  }

  template <typename Poly, typename Domain>
  void OpenFixedColumns(
      const Domain* domain, const F& x,
      const ConstraintSystem<F>& constraint_system,
      std::vector<crypto::PolynomialOpening<Poly, C>>& openings) const {
    OpenColumns(domain, x, constraint_system.fixed_queries(),
                data_.fixed_commitments, data_.fixed_evals, openings);
  }

  template <typename Poly>
  void Open(const F& x, const F& x_n, C& h_commitment, const F& h_eval,
            std::vector<crypto::PolynomialOpening<Poly, C>>& openings) const {
    // TODO(chokobole): Remove |ToAffine()| since this assumes commitment is an
    // elliptic curve point.
    h_commitment = C::template LinearCombination</*forward=*/false>(
                       data_.h_poly_commitments, x_n)
                       .ToAffine();

#define OPENING(commitment, point, eval) \
  base::Ref<const C>(&commitment), point, eval

    openings.emplace_back(OPENING(h_commitment, x, h_eval));
    openings.emplace_back(
        OPENING(data_.random_poly_commitment, x, data_.random_eval));

#undef OPENING
  }

 private:
  template <typename Poly, typename Domain, ColumnType CType>
  static void OpenColumns(
      const Domain* domain, const F& x,
      const std::vector<QueryData<CType>>& queries,
      absl::Span<const C> commitments, absl::Span<const F> evals,
      std::vector<crypto::PolynomialOpening<Poly, C>>& openings) {
#define OPENING(commitment, point, eval) \
  base::Ref<const C>(&commitment), point, eval

    for (size_t i = 0; i < queries.size(); ++i) {
      const QueryData<CType>& query = queries[i];
      const ColumnKey<CType>& column = query.column();
      F point = query.rotation().RotateOmega(domain, x);
      openings.emplace_back(
          OPENING(commitments[column.index()], point, evals[i]));
    }

#undef OPENING
  }

  const VanishingVerifierData<F, C>& data_;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_VANISHING_VANISHING_VERIFIER_H_
