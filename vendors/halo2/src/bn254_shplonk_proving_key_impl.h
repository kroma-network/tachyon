#ifndef VENDORS_HALO2_SRC_BN254_SHPLONK_PROVING_KEY_IMPL_H_
#define VENDORS_HALO2_SRC_BN254_SHPLONK_PROVING_KEY_IMPL_H_

#include "tachyon/math/elliptic_curves/bn/bn254/bn254.h"
#include "tachyon/zk/base/commitments/shplonk_extension.h"
#include "vendors/halo2/src/degrees.h"
#include "vendors/halo2/src/shplonk_proving_key_impl.h"

namespace tachyon::halo2_api::bn254 {

using PCS =
    zk::SHPlonkExtension<math::bn254::BN254Curve, kMaxDegree,
                         kMaxExtendedDegree, math::bn254::G1AffinePoint>;

class SHPlonkProvingKeyImpl : public ProvingKeyImpl<PCS> {
 public:
  explicit SHPlonkProvingKeyImpl(rust::Slice<const uint8_t> bytes)
      : ProvingKeyImpl<PCS>(bytes) {}
};

}  // namespace tachyon::halo2_api::bn254

#endif  // VENDORS_HALO2_SRC_BN254_SHPLONK_PROVING_KEY_IMPL_H_
