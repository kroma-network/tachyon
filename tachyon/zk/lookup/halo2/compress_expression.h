// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_LOOKUP_HALO2_COMPRESS_EXPRESSION_H_
#define TACHYON_ZK_LOOKUP_HALO2_COMPRESS_EXPRESSION_H_

#include <memory>
#include <utility>
#include <vector>

#include "tachyon/base/compiler_specific.h"
#include "tachyon/base/openmp_util.h"
#include "tachyon/zk/lookup/proving_evaluator.h"

namespace tachyon::zk::lookup::halo2 {

template <typename Domain, typename Evals, typename F>
Evals CompressExpressions(
    const Domain* domain,
    const std::vector<std::unique_ptr<Expression<F>>>& expressions,
    const F& theta, const ProvingEvaluator<Evals>& evaluator_tpl) {
  Evals compressed_evals = domain->template Zero<Evals>();
  std::vector<F>& compressed_values = compressed_evals.evaluations();

  for (size_t expr_idx = 0; expr_idx < expressions.size(); ++expr_idx) {
    if (UNLIKELY(expr_idx == 0)) {
      OPENMP_PARALLEL_FOR(size_t i = 0; i < compressed_values.size(); ++i) {
        ProvingEvaluator<Evals> evaluator = evaluator_tpl;
        evaluator.set_idx(i);
        compressed_values[i] = evaluator.Evaluate(expressions[expr_idx].get());
      }
    } else {
      OPENMP_PARALLEL_FOR(size_t i = 0; i < compressed_values.size(); ++i) {
        ProvingEvaluator<Evals> evaluator = evaluator_tpl;
        evaluator.set_idx(i);
        compressed_values[i] *= theta;
        compressed_values[i] += evaluator.Evaluate(expressions[expr_idx].get());
      }
    }
  }
  return compressed_evals;
}

}  // namespace tachyon::zk::lookup::halo2

#endif  // TACHYON_ZK_LOOKUP_HALO2_COMPRESS_EXPRESSION_H_
