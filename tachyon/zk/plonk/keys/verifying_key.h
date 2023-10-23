#ifndef TACHYON_ZK_PLONK_KEYS_VERIFYING_KEY_H_
#define TACHYON_ZK_PLONK_KEYS_VERIFYING_KEY_H_

#include <stddef.h>

#include <memory>
#include <vector>

#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain.h"
#include "tachyon/zk/plonk/constraint_system.h"

namespace tachyon::zk {

template <typename Curve, size_t MaxDegree>
struct VerifyingKey {
  using ScalarField = typename math::AffinePoint<Curve>::ScalarField;

  std::unique_ptr<math::UnivariateEvaluationDomain<ScalarField, MaxDegree>>
      domain;
  std::vector<math::AffinePoint<Curve>> fixed_commitments;
  ConstraintSystem<ScalarField> cs;
  // Cached maximum degree of |cs| (which doesn't change after
  // construction).
  size_t cs_degree;
  // The representative of this |VerifyingKey| in transcripts.
  ScalarField transcript_repr;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_KEYS_VERIFYING_KEY_H_
