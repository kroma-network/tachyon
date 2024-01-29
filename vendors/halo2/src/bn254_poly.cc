#include "vendors/halo2/include/bn254_poly.h"

#include "vendors/halo2/src/bn254_poly_impl.h"

namespace tachyon::halo2_api::bn254 {

Poly::Poly() : impl_(new PolyImpl()) {}

}  // namespace tachyon::halo2_api::bn254
