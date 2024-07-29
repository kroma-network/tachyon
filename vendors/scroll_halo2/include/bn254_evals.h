#ifndef VENDORS_SCROLL_HALO2_INCLUDE_BN254_EVALS_H_
#define VENDORS_SCROLL_HALO2_INCLUDE_BN254_EVALS_H_

#include <stddef.h>

#include <memory>
#include <utility>

#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluations.h"

namespace tachyon::halo2_api::bn254 {

struct Fr;

class Evals {
 public:
  Evals();
  explicit Evals(tachyon_bn254_univariate_evaluations* evals) : evals_(evals) {}
  Evals(const Evals& other) = delete;
  Evals& operator=(const Evals& other) = delete;
  ~Evals();

  tachyon_bn254_univariate_evaluations* evals() { return evals_; }
  const tachyon_bn254_univariate_evaluations* evals() const { return evals_; }

  tachyon_bn254_univariate_evaluations* release() {
    return std::exchange(evals_, nullptr);
  }

  size_t len() const;
  void set_value(size_t idx, const Fr& value);
  std::unique_ptr<Evals> clone() const;

 private:
  tachyon_bn254_univariate_evaluations* evals_;
};

std::unique_ptr<Evals> zero_evals();

}  // namespace tachyon::halo2_api::bn254

#endif  // VENDORS_SCROLL_HALO2_INCLUDE_BN254_EVALS_H_
