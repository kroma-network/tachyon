#ifndef VENDORS_HALO2_INCLUDE_BN254_RATIONAL_EVALS_VIEW_H_
#define VENDORS_HALO2_INCLUDE_BN254_RATIONAL_EVALS_VIEW_H_

#include <stddef.h>

#include "tachyon/c/math/polynomials/univariate/bn254_univariate_rational_evaluations.h"

namespace tachyon::halo2_api::bn254 {

struct Fr;

class RationalEvalsView {
 public:
  RationalEvalsView(tachyon_bn254_univariate_rational_evaluations* evals,
                    size_t start, size_t len);
  RationalEvalsView(const RationalEvalsView& other) = delete;
  RationalEvalsView& operator=(const RationalEvalsView& other) = delete;
  ~RationalEvalsView() = default;

  void set_zero(size_t idx);
  void set_trivial(size_t idx, const Fr& numerator);
  void set_rational(size_t idx, const Fr& numerator, const Fr& denominator);
  void evaluate(size_t idx, Fr& value) const;

 private:
  // not owned
  tachyon_bn254_univariate_rational_evaluations* const evals_;
  const size_t start_ = 0;
  const size_t len_ = 0;
};

}  // namespace tachyon::halo2_api::bn254

#endif  // VENDORS_HALO2_INCLUDE_BN254_RATIONAL_EVALS_VIEW_H_
