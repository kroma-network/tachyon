#ifndef TACHYON_ZK_BASE_VERIFIER_QUERY_H_
#define TACHYON_ZK_BASE_VERIFIER_QUERY_H_

namespace tachyon::zk {

template <typename PCSTy>
class VerifierQuery {
 public:
  using F = typename PCSTy::Field;
  using Commitment = typename PCSTy::Commitment;

  VerifierQuery(const F& point, const Commitment& commitment, const F& eval)
      : point_(point), commitment_(commitment), eval_(eval) {}

  const F& point() const { return point_; }
  const Commitment& commitment() const { return commitment_; }
  const F& eval() const { return eval_; }

 private:
  const F& point_;
  const Commitment& commitment_;
  const F& eval_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_VERIFIER_QUERY_H_
