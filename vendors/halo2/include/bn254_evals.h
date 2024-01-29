#ifndef VENDORS_HALO2_INCLUDE_BN254_EVALS_H_
#define VENDORS_HALO2_INCLUDE_BN254_EVALS_H_

#include <stddef.h>

#include <memory>

namespace tachyon::halo2_api::bn254 {

struct Fr;
class EvalsImpl;

class Evals {
 public:
  Evals();

  EvalsImpl* impl() { return impl_.get(); }
  const EvalsImpl* impl() const { return impl_.get(); }

  size_t len() const;
  void set_value(size_t idx, const Fr& value);
  std::unique_ptr<Evals> clone() const;

 private:
  std::shared_ptr<EvalsImpl> impl_;
};

std::unique_ptr<Evals> zero_evals();

}  // namespace tachyon::halo2_api::bn254

#endif  // VENDORS_HALO2_INCLUDE_BN254_EVALS_H_
