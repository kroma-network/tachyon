#ifndef TACHYON_ZK_BASE_PROVER_QUERY_H_
#define TACHYON_ZK_BASE_PROVER_QUERY_H_

#include "tachyon/zk/base/ref_aliases.h"

namespace tachyon::zk {

template <typename PCSTy>
class ProverQuery {
 public:
  using F = typename PCSTy::Field;
  using Poly = typename PCSTy::Poly;

  ProverQuery(PointRef<const F> point, BlindedPolyRef<const Poly> poly)
      : point_(point), poly_(poly) {}

  PointRef<const F> point() const { return point_; }
  BlindedPolyRef<const Poly>& poly() const { return poly_; }

  F Evaluate() const { return poly_->poly().Evaluate(*point_); }

 private:
  PointRef<const F> point_;
  BlindedPolyRef<const Poly> poly_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_PROVER_QUERY_H_
