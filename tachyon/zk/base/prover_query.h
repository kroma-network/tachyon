#ifndef TACHYON_ZK_BASE_PROVER_QUERY_H_
#define TACHYON_ZK_BASE_PROVER_QUERY_H_

#include "tachyon/zk/base/polynomial_pointer.h"

namespace tachyon::zk {

template <typename PCSTy>
class ProverQuery {
 public:
  using F = typename PCSTy::Field;
  using Poly = typename PCSTy::Poly;

  ProverQuery(const F& point, const Poly& poly, const F& blind)
      : point_(point), poly_(poly), blind_(blind) {}

  const F& point() const { return point_; }
  const Poly& poly() const { return poly_; }
  const F& blind() const { return blind_; }

  F Evaluate() const { return poly_.Evaluate(point_); }

  PolynomialPointer<PCSTy> GetCommitment() const {
    return PolynomialPointer<PCSTy>(poly_, blind_);
  }

 private:
  const F& point_;
  const Poly& poly_;
  const F& blind_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_PROVER_QUERY_H_
