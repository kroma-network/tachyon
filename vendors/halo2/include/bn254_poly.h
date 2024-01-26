#ifndef VENDORS_HALO2_INCLUDE_BN254_POLY_H_
#define VENDORS_HALO2_INCLUDE_BN254_POLY_H_

#include <memory>

namespace tachyon::halo2_api::bn254 {

class PolyImpl;

class Poly {
 public:
  Poly();

  PolyImpl* impl() { return impl_.get(); }

 private:
  std::shared_ptr<PolyImpl> impl_;
};

}  // namespace tachyon::halo2_api::bn254

#endif  // VENDORS_HALO2_INCLUDE_BN254_POLY_H_
