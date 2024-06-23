#include "tachyon/c/zk/plonk/keys/bn254_plonk_proving_key.h"

#include "gtest/gtest.h"

#include "tachyon/c/zk/plonk/keys/bn254_plonk_proving_key_type_traits.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::zk::plonk {

namespace {

class Bn254PlonkProvingKeyTest : public math::FiniteFieldTest<math::bn254::Fr> {
};

}  // namespace

TEST_F(Bn254PlonkProvingKeyTest, GetVerifyingKey) {
  c::zk::plonk::bn254::ProvingKeyImpl cpp_pkey;

  tachyon_bn254_plonk_proving_key* pkey = c::base::c_cast(&cpp_pkey);
  EXPECT_EQ(tachyon_bn254_plonk_proving_key_get_verifying_key(pkey),
            reinterpret_cast<const tachyon_bn254_plonk_verifying_key*>(
                &cpp_pkey.verifying_key()));
}

}  // namespace tachyon::zk::plonk
