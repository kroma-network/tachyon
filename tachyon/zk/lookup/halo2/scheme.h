#ifndef TACHYON_ZK_LOOKUP_HALO2_SCHEME_H_
#define TACHYON_ZK_LOOKUP_HALO2_SCHEME_H_

#include "tachyon/zk/lookup/halo2/prover.h"
#include "tachyon/zk/lookup/type.h"

namespace tachyon::zk::lookup::halo2 {

template <typename _Poly, typename _Evals, typename _Commitment,
          typename _ExtendedPoly, typename _ExtendedEvals>
struct Scheme {
  using Poly = _Poly;
  using Evals = _Evals;
  using Commitment = _Commitment;
  using ExtendedPoly = _ExtendedPoly;
  using ExtendedEvals = _ExtendedEvals;
  using Field = typename Poly::Field;

  using Prover = lookup::halo2::Prover<Poly, Evals>;

  constexpr static Type kType = Type::kHalo2;
};

}  // namespace tachyon::zk::lookup::halo2

#endif  // TACHYON_ZK_LOOKUP_HALO2_SCHEME_H_
