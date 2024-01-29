#ifndef VENDORS_HALO2_INCLUDE_BN254_RATIONAL_EVALS_H_
#define VENDORS_HALO2_INCLUDE_BN254_RATIONAL_EVALS_H_

#include <stddef.h>

#include <memory>

namespace tachyon::halo2_api::bn254 {

struct Fr;
class RationalEvalsImpl;

class RationalEvals {
 public:
  RationalEvals();

  RationalEvalsImpl* impl() { return impl_.get(); }
  const RationalEvalsImpl* impl() const { return impl_.get(); }

  size_t len() const;
  void set_zero(size_t idx);
  void set_trivial(size_t idx, const Fr& numerator);
  void set_rational(size_t idx, const Fr& numerator, const Fr& denominator);
  std::unique_ptr<RationalEvals> clone() const;

 private:
  std::shared_ptr<RationalEvalsImpl> impl_;
};

}  // namespace tachyon::halo2_api::bn254

#endif  // VENDORS_HALO2_INCLUDE_BN254_RATIONAL_EVALS_H_
