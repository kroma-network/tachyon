#ifndef TACHYON_ZK_PLONK_HALO2_BN254_SHPLONK_PROVER_TEST_H_
#define TACHYON_ZK_PLONK_HALO2_BN254_SHPLONK_PROVER_TEST_H_

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/bn254.h"
#include "tachyon/math/elliptic_curves/bn/bn254/halo2/bn254.h"
#include "tachyon/zk/base/commitments/shplonk_extension.h"
#include "tachyon/zk/lookup/halo2/scheme.h"
#include "tachyon/zk/plonk/halo2/prover_test.h"

namespace tachyon::zk::plonk::halo2 {

using PCS = SHPlonkExtension<math::bn254::BN254Curve, kMaxDegree,
                             kMaxExtendedDegree, math::bn254::G1AffinePoint>;
using LS =
    lookup::halo2::Scheme<typename PCS::Poly, typename PCS::Evals,
                          typename PCS::Commitment, typename PCS::ExtendedPoly,
                          typename PCS::ExtendedEvals>;

class BN254SHPlonkProverTest : public ProverTest<PCS, LS> {
 public:
  static void SetUpTestSuite() {
    math::bn254::BN254Curve::Init();
    math::halo2::OverrideSubgroupGenerator();
  }
};

}  // namespace tachyon::zk::plonk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_BN254_SHPLONK_PROVER_TEST_H_
