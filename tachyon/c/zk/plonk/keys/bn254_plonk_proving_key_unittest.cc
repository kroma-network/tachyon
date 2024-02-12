#include "tachyon/c/zk/plonk/keys/bn254_plonk_proving_key.h"

#include "gtest/gtest.h"

#include "tachyon/c/math/polynomials/constants.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/zk/plonk/keys/proving_key.h"

namespace tachyon::zk::plonk {

namespace {

class Bn254PlonkProvingKeyTest : public math::FiniteFieldTest<math::bn254::Fr> {
 public:
  using Poly =
      math::UnivariateDensePolynomial<math::bn254::Fr, c::math::kMaxDegree>;
  using Evals =
      math::UnivariateEvaluations<math::bn254::Fr, c::math::kMaxDegree>;
};

}  // namespace

TEST_F(Bn254PlonkProvingKeyTest, GetVerifyingKey) {
  ProvingKey<Poly, Evals, math::bn254::G1AffinePoint> cpp_pkey;

  tachyon_bn254_plonk_proving_key* pkey =
      reinterpret_cast<tachyon_bn254_plonk_proving_key*>(&cpp_pkey);
  EXPECT_EQ(tachyon_bn254_plonk_proving_key_get_verifying_key(pkey),
            reinterpret_cast<const tachyon_bn254_plonk_verifying_key*>(
                &cpp_pkey.verifying_key()));
}

}  // namespace tachyon::zk::plonk
