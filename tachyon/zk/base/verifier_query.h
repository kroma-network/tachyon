#ifndef TACHYON_ZK_BASE_VERIFIER_QUERY_H_
#define TACHYON_ZK_BASE_VERIFIER_QUERY_H_

#include "tachyon/zk/base/ref_aliases.h"

namespace tachyon::zk {

template <typename PCSTy>
class VerifierQuery {
 public:
  using F = typename PCSTy::Field;
  using Commitment = typename PCSTy::Commitment;

  VerifierQuery(PointRef<const F> point,
                CommitmentRef<const Commitment> commitment,
                FieldRef<const F> evaluated)
      : point_(point), commitment_(commitment), evaluated_(evaluated) {}

  PointRef<const F> point() const { return point_; }
  CommitmentRef<const Commitment> commitment() const { return commitment_; }
  FieldRef<const F> evaluated() const { return evaluated_; }

 private:
  PointRef<const F> point_;
  CommitmentRef<const Commitment> commitment_;
  FieldRef<const F> evaluated_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_VERIFIER_QUERY_H_
