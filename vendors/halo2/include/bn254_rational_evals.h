#ifndef VENDORS_HALO2_INCLUDE_BN254_RATIONAL_EVALS_H_
#define VENDORS_HALO2_INCLUDE_BN254_RATIONAL_EVALS_H_

#include <stddef.h>

#include <memory>
#include <utility>

#include "tachyon/c/math/polynomials/univariate/bn254_univariate_rational_evaluations.h"

namespace tachyon::halo2_api::bn254 {

struct Fr;

class RationalEvals {
 public:
  RationalEvals();
  explicit RationalEvals(tachyon_bn254_univariate_rational_evaluations* evals)
      : evals_(evals) {}
  RationalEvals(const RationalEvals& other) = delete;
  RationalEvals& operator=(const RationalEvals& other) = delete;
  ~RationalEvals();

  tachyon_bn254_univariate_rational_evaluations* evals() { return evals_; }
  const tachyon_bn254_univariate_rational_evaluations* evals() const {
    return evals_;
  }

  tachyon_bn254_univariate_rational_evaluations* release() {
    return std::exchange(evals_, nullptr);
  }

  size_t len() const;
  void set_zero(size_t idx);
  void set_trivial(size_t idx, const Fr& numerator);
  void set_rational(size_t idx, const Fr& numerator, const Fr& denominator);
  std::unique_ptr<RationalEvals> clone() const;

 private:
  tachyon_bn254_univariate_rational_evaluations* evals_;
};

}  // namespace tachyon::halo2_api::bn254

#endif  // VENDORS_HALO2_INCLUDE_BN254_RATIONAL_EVALS_H_
