// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_EXPRESSIONS_EVALUATOR_H_
#define TACHYON_ZK_EXPRESSIONS_EVALUATOR_H_

#include "tachyon/zk/expressions/expression.h"

namespace tachyon::zk {

template <typename F, typename T>
class Evaluator {
 public:
  virtual ~Evaluator() = default;

  virtual T Evaluate(const Expression<F>* input) = 0;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_EXPRESSIONS_EVALUATOR_H_
