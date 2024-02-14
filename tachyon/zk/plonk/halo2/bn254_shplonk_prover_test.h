#ifndef TACHYON_ZK_PLONK_HALO2_BN254_SHPLONK_PROVER_TEST_H_
#define TACHYON_ZK_PLONK_HALO2_BN254_SHPLONK_PROVER_TEST_H_

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/bn254.h"
#include "tachyon/zk/base/commitments/shplonk_extension.h"
#include "tachyon/zk/plonk/halo2/prover_test.h"

namespace tachyon::zk::plonk::halo2 {

using PCS = SHPlonkExtension<math::bn254::BN254Curve, kMaxDegree,
                             kMaxExtendedDegree, math::bn254::G1AffinePoint>;

class BN254SHPlonkProverTest : public ProverTest<PCS> {
 public:
  static void SetUpTestSuite() { math::bn254::BN254Curve::Init(); }
};

}  // namespace tachyon::zk::plonk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_BN254_SHPLONK_PROVER_TEST_H_
