#ifndef TACHYON_C_ZK_PLONK_HALO2_BN254_PS_H_
#define TACHYON_C_ZK_PLONK_HALO2_BN254_PS_H_

#include "tachyon/c/math/polynomials/constants.h"
#include "tachyon/math/elliptic_curves/bn/bn254/bn254.h"
#include "tachyon/zk/base/commitments/gwc_extension.h"
#include "tachyon/zk/base/commitments/shplonk_extension.h"
#include "tachyon/zk/plonk/halo2/proving_scheme.h"

namespace tachyon::c::zk::plonk::halo2::bn254 {

using GWCPCS = tachyon::zk::GWCExtension<tachyon::math::bn254::BN254Curve,
                                         math::kMaxDegree, math::kMaxDegree,
                                         tachyon::math::bn254::G1AffinePoint>;

using SHPlonkPCS =
    tachyon::zk::SHPlonkExtension<tachyon::math::bn254::BN254Curve,
                                  math::kMaxDegree, math::kMaxDegree,
                                  tachyon::math::bn254::G1AffinePoint>;

using PSEGWC = tachyon::zk::plonk::halo2::ProvingScheme<
    tachyon::zk::plonk::halo2::Vendor::kPSE, tachyon::zk::lookup::Type::kHalo2,
    GWCPCS>;

using PSESHPlonk = tachyon::zk::plonk::halo2::ProvingScheme<
    tachyon::zk::plonk::halo2::Vendor::kPSE, tachyon::zk::lookup::Type::kHalo2,
    SHPlonkPCS>;

using ScrollGWC = tachyon::zk::plonk::halo2::ProvingScheme<
    tachyon::zk::plonk::halo2::Vendor::kScroll,
    tachyon::zk::lookup::Type::kLogDerivativeHalo2, GWCPCS>;

using ScrollSHPlonk = tachyon::zk::plonk::halo2::ProvingScheme<
    tachyon::zk::plonk::halo2::Vendor::kScroll,
    tachyon::zk::lookup::Type::kLogDerivativeHalo2, SHPlonkPCS>;

}  // namespace tachyon::c::zk::plonk::halo2::bn254

#endif  // TACHYON_C_ZK_PLONK_HALO2_BN254_PS_H_
