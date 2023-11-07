#ifndef TACHYON_ZK_BASE_VERIFIER_QUERY_H_
#define TACHYON_ZK_BASE_VERIFIER_QUERY_H_

#include "tachyon/zk/base/ref.h"

namespace tachyon::zk {

template <typename PCSTy>
class VerifierQuery {
 public:
  using F = typename PCSTy::Field;
  using Commitment = typename PCSTy::Commitment;

  VerifierQuery(Ref<const F> point, Ref<const Commitment> commitment,
                Ref<const F> evaluated)
      : point_(point), commitment_(commitment), evaluated_(evaluated) {}

  Ref<const F> point() const { return point_; }
  Ref<const Commitment> commitment() const { return commitment_; }
  Ref<const F> evaluated() const { return evaluated_; }

 private:
  Ref<const F> point_;
  Ref<const Commitment> commitment_;
  Ref<const F> evaluated_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_VERIFIER_QUERY_H_
