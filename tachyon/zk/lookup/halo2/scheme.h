#ifndef TACHYON_ZK_LOOKUP_HALO2_SCHEME_H_
#define TACHYON_ZK_LOOKUP_HALO2_SCHEME_H_

#include "tachyon/zk/lookup/halo2/evaluator.h"
#include "tachyon/zk/lookup/halo2/prover.h"
#include "tachyon/zk/lookup/halo2/verifier.h"
#include "tachyon/zk/lookup/halo2/verifier_data.h"
#include "tachyon/zk/lookup/type.h"

namespace tachyon::zk::lookup::halo2 {

template <typename _Poly, typename _Evals, typename _Commitment>
struct Scheme {
  using Poly = _Poly;
  using Evals = _Evals;
  using Commitment = _Commitment;
  using Field = typename Poly::Field;

  using Prover = lookup::halo2::Prover<Poly, Evals>;
  using Verifier = lookup::halo2::Verifier<Field, Commitment>;
  using VerifierData = lookup::halo2::VerifierData<Field, Commitment>;
  using Evaluator = lookup::halo2::Evaluator<Field, Evals>;

  constexpr static Type type = Type::kHalo2;
};

}  // namespace tachyon::zk::lookup::halo2

#endif  // TACHYON_ZK_LOOKUP_HALO2_SCHEME_H_
