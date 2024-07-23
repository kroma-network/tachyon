// Copyright (c) 2022-2024 Scroll
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.scroll and the LICENCE-APACHE.scroll
// file.

#ifndef TACHYON_ZK_LOOKUP_LOG_DERIVATIVE_HALO2_VERIFIER_H_
#define TACHYON_ZK_LOOKUP_LOG_DERIVATIVE_HALO2_VERIFIER_H_

#include <memory>
#include <vector>

#include "tachyon/crypto/commitments/polynomial_openings.h"
#include "tachyon/zk/lookup/argument.h"
#include "tachyon/zk/lookup/halo2/opening_point_set.h"
#include "tachyon/zk/lookup/log_derivative_halo2/verifier_data.h"
#include "tachyon/zk/lookup/verifier.h"
#include "tachyon/zk/plonk/base/l_values.h"
#include "tachyon/zk/plonk/expressions/verifying_evaluator.h"
#include "tachyon/zk/plonk/halo2/proof.h"

namespace tachyon::zk::lookup {
namespace log_derivative_halo2 {

template <typename F, typename C>
class Verifier final
    : public lookup::Verifier<typename log_derivative_halo2::Verifier<F, C>> {
 public:
  using Proof = plonk::halo2::LogDerivativeHalo2Proof<F, C>;

  Verifier(const Proof& proof, size_t circuit_idx)
      : data_(proof.ToLookupVerifierData(circuit_idx)) {}

  Verifier(const Proof& proof, size_t circuit_idx,
           const plonk::LValues<F>& l_values)
      : data_(proof.ToLookupVerifierData(circuit_idx)), l_values_(&l_values) {}

  void DoEvaluate(const std::vector<Argument<F>>& arguments,
                  std::vector<F>& evals) {
    plonk::VerifyingEvaluator<F> evaluator(data_);

    F active_rows = F::One() - (l_values_->last + l_values_->blind);

    for (size_t i = 0; i < data_.grand_sum_commitments.size(); ++i) {
      // l_first(X) * ϕ(X) = 0
      evals.push_back(l_values_->first * data_.grand_sum_evals[i]);
      // l_last(X) * ϕ(X) = 0
      evals.push_back(l_values_->last * data_.grand_sum_evals[i]);
      // (1 - (l_last(X) + l_blind(X))) * (lhs - rhs) = 0
      evals.push_back(active_rows *
                      CreateGrandSumEvaluation(i, arguments[i], evaluator));
    }
  }

  template <typename Poly>
  void DoOpen(const halo2::OpeningPointSet<F>& point_set,
              std::vector<crypto::PolynomialOpening<Poly, C>>& openings) const {
    if (data_.grand_sum_commitments.empty()) return;

#define OPENING(commitment, point, eval) \
  base::Ref<const C>(&data_.commitment), point_set.point, data_.eval

    for (size_t i = 0; i < data_.grand_sum_commitments.size(); ++i) {
      openings.emplace_back(
          OPENING(grand_sum_commitments[i], x, grand_sum_evals[i]));
      openings.emplace_back(
          OPENING(grand_sum_commitments[i], x_next, grand_sum_next_evals[i]));
      openings.emplace_back(OPENING(m_poly_commitments[i], x, m_evals[i]));
    }

#undef OPENING
  }

 private:
  F CompressExpressions(
      const std::vector<std::unique_ptr<Expression<F>>>& expressions,
      plonk::VerifyingEvaluator<F>& evaluator) const {
    F compressed_value = F::Zero();
    for (const std::unique_ptr<Expression<F>>& expression : expressions) {
      compressed_value *= data_.theta;
      compressed_value += evaluator.Evaluate(expression.get());
    }
    return compressed_value;
  }

  F CreateGrandSumEvaluation(size_t i, const Argument<F>& argument,
                             plonk::VerifyingEvaluator<F>& evaluator) {
    // φᵢ(X) = fᵢ(X) + β
    std::vector<F> f_evals = base::Map(
        argument.inputs_expressions(),
        [this, &evaluator](const std::vector<std::unique_ptr<Expression<F>>>&
                               input_expressions) {
          return CompressExpressions(input_expressions, evaluator) + data_.beta;
        });

    F t_eval = CompressExpressions(argument.table_expressions(), evaluator);

    F tau = t_eval + data_.beta;

    // Π(φᵢ(X))
    F prod_fi =
        std::accumulate(f_evals.begin(), f_evals.end(), F::One(),
                        [](F& acc, const F& f_eval) { return acc *= f_eval; });

    CHECK(F::BatchInverseInPlace(f_evals));

    // Σ 1/(φᵢ(X))
    F sum_inv_fi =
        std::accumulate(f_evals.begin(), f_evals.end(), F::Zero(),
                        [](F& acc, const F& f_eval) { return acc += f_eval; });

    // LHS = τ(X) * Π(φᵢ(X)) * (ϕ(ω * X) - ϕ(X))
    F lhs = tau * prod_fi *
            (data_.grand_sum_next_evals[i] - data_.grand_sum_evals[i]);

    // RHS = τ(X) * Π(φᵢ(X)) * (Σ 1/(φᵢ(X)) - m(X) / τ(X))
    F rhs = tau * prod_fi * (sum_inv_fi - data_.m_evals[i] * *tau.Inverse());

    return lhs - rhs;
  }

  VerifierData<F, C> data_;
  const plonk::LValues<F>* l_values_ = nullptr;
};

}  // namespace log_derivative_halo2

template <typename F, typename C>
struct VerifierTraits<log_derivative_halo2::Verifier<F, C>> {
  using Field = F;
};

}  // namespace tachyon::zk::lookup

#endif  // TACHYON_ZK_LOOKUP_LOG_DERIVATIVE_HALO2_VERIFIER_H_
