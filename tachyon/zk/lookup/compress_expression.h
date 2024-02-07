// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_LOOKUP_COMPRESS_EXPRESSION_H_
#define TACHYON_ZK_LOOKUP_COMPRESS_EXPRESSION_H_

#include <memory>
#include <utility>
#include <vector>

#include "tachyon/zk/expressions/evaluator/simple_evaluator.h"

namespace tachyon::zk {

template <typename Domain, typename Evals, typename F>
Evals CompressExpressions(
    const Domain* domain,
    const std::vector<std::unique_ptr<Expression<F>>>& expressions,
    const F& theta, const SimpleEvaluator<Evals>& evaluator_tpl) {
  Evals compressed_value = domain->template Zero<Evals>();
  Evals values = domain->template Zero<Evals>();

  for (size_t expr_idx = 0; expr_idx < expressions.size(); ++expr_idx) {
    base::Parallelize(
        values.evaluations(),
        [expr_idx, &expressions, &evaluator_tpl](
            absl::Span<F> chunk, size_t chunk_index, size_t chunk_size) {
          size_t start = chunk_index * chunk_size;
          for (size_t i = 0; i < chunk.size(); ++i) {
            SimpleEvaluator<Evals> evaluator = evaluator_tpl;
            evaluator.set_idx(start + i);
            chunk[i] = evaluator.Evaluate(expressions[expr_idx].get());
          }
        });
    compressed_value *= theta;
    compressed_value += values;
  }
  return compressed_value;
}

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_LOOKUP_COMPRESS_EXPRESSION_H_
