#ifndef TACHYON_C_ZK_PLONK_HALO2_BN254_GWC_PCS_H_
#define TACHYON_C_ZK_PLONK_HALO2_BN254_GWC_PCS_H_

#include "tachyon/c/math/polynomials/constants.h"
#include "tachyon/math/elliptic_curves/bn/bn254/bn254.h"
#include "tachyon/zk/base/commitments/gwc_extension.h"

namespace tachyon::c::zk::plonk::halo2::bn254 {

using GWCPCS = tachyon::zk::GWCExtension<tachyon::math::bn254::BN254Curve,
                                         math::kMaxDegree, math::kMaxDegree,
                                         tachyon::math::bn254::G1AffinePoint>;

}  // namespace tachyon::c::zk::plonk::halo2::bn254

#endif  // TACHYON_C_ZK_PLONK_HALO2_BN254_GWC_PCS_H_
