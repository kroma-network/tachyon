// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_LOOKUP_LOOKUP_ARGUMENT_RUNNER_H_
#define TACHYON_ZK_PLONK_LOOKUP_LOOKUP_ARGUMENT_RUNNER_H_

#include "tachyon/zk/base/prover.h"
#include "tachyon/zk/plonk/circuit/expressions/evaluator/simple_evaluator.h"
#include "tachyon/zk/plonk/lookup/lookup_argument.h"
#include "tachyon/zk/plonk/lookup/lookup_permuted.h"

namespace tachyon::zk {

template <typename Poly, typename Evals>
class LookupArgumentRunner {
 public:
  LookupArgumentRunner() = delete;

  template <typename PCSTy, typename ExtendedDomain, typename F>
  static LookupPermuted<Poly, Evals> PermuteArgument(
      Prover<PCSTy, ExtendedDomain>* prover, const LookupArgument<F>& argument,
      const F& theta, const SimpleEvaluator<Evals>& evaluator_tpl);
};

}  // namespace tachyon::zk

#include "tachyon/zk/plonk/lookup/lookup_argument_runner_impl.h"

#endif  // TACHYON_ZK_PLONK_LOOKUP_LOOKUP_ARGUMENT_RUNNER_H_
