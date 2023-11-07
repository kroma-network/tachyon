#ifndef TACHYON_ZK_BASE_PROVER_QUERY_H_
#define TACHYON_ZK_BASE_PROVER_QUERY_H_

#include "tachyon/zk/base/blinded_polynomial.h"

namespace tachyon::zk {

template <typename PCSTy>
class ProverQuery {
 public:
  using F = typename PCSTy::Field;
  using Poly = typename PCSTy::Poly;

  ProverQuery(Ref<const F> point, Ref<const BlindedPolynomial<Poly>> poly)
      : point_(point), poly_(poly) {}

  Ref<const F> point() const { return point_; }
  Ref<const BlindedPolynomial<Poly>> poly() const { return poly_; }

  F Evaluate() const { return poly_->poly().Evaluate(*point_); }

 private:
  Ref<const F> point_;
  Ref<const BlindedPolynomial<Poly>> poly_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_PROVER_QUERY_H_
