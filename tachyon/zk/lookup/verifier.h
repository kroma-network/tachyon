#ifndef TACHYON_ZK_LOOKUP_VERIFIER_H_
#define TACHYON_ZK_LOOKUP_VERIFIER_H_

#include <memory>
#include <vector>

#include "tachyon/zk/lookup/lookup_argument.h"
#include "tachyon/zk/lookup/verifier_traits_forward.h"

namespace tachyon::zk::lookup {

template <typename Derived>
class Verifier {
 public:
  using Field = typename VerifierTraits<Derived>::Field;

  void Evaluate(const std::vector<Argument<Field>>& arguments,
                std::vector<Field>& evals) {
    Derived* derived = static_cast<Derived*>(this);
    return derived->DoEvaluate(arguments, evals);
  }

  template <typename OpeningPointSet, typename Openings>
  void Open(const OpeningPointSet& point_set, Openings& openings) {
    Derived* derived = static_cast<Derived*>(this);
    return derived->DoOpen(point_set, openings);
  }
};

}  // namespace tachyon::zk::lookup

#endif  // TACHYON_ZK_LOOKUP_VERIFIER_H_
