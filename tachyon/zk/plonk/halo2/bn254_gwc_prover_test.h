#ifndef TACHYON_ZK_PLONK_HALO2_BN254_GWC_PROVER_TEST_H_
#define TACHYON_ZK_PLONK_HALO2_BN254_GWC_PROVER_TEST_H_

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/bn254.h"
#include "tachyon/math/elliptic_curves/bn/bn254/halo2/bn254.h"
#include "tachyon/zk/base/commitments/gwc_extension.h"
#include "tachyon/zk/plonk/halo2/prover_test.h"
#include "tachyon/zk/plonk/halo2/proving_scheme.h"

namespace tachyon::zk::plonk::halo2 {

using PCS = GWCExtension<math::bn254::BN254Curve, kMaxDegree,
                         kMaxExtendedDegree, math::bn254::G1AffinePoint>;
using PS = ProvingScheme<Vendor::kScroll, lookup::Type::kHalo2, PCS>;

class BN254GWCProverTest : public ProverTest<PS> {
 public:
  static void SetUpTestSuite() {
    math::bn254::BN254Curve::Init();
    math::halo2::OverrideSubgroupGenerator();
  }
};

}  // namespace tachyon::zk::plonk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_BN254_GWC_PROVER_TEST_H_
