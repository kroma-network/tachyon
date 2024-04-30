#include "tachyon/c/zk/plonk/keys/bn254_plonk_proving_key.h"

#include "gtest/gtest.h"

#include "tachyon/c/zk/plonk/halo2/bn254_ls.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/zk/plonk/keys/proving_key.h"

namespace tachyon::zk::plonk {

namespace {

class Bn254PlonkProvingKeyTest : public math::FiniteFieldTest<math::bn254::Fr> {
 public:
  using LS = c::zk::plonk::halo2::bn254::LS;
};

}  // namespace

TEST_F(Bn254PlonkProvingKeyTest, GetVerifyingKey) {
  ProvingKey<LS> cpp_pkey;

  tachyon_bn254_plonk_proving_key* pkey =
      reinterpret_cast<tachyon_bn254_plonk_proving_key*>(&cpp_pkey);
  EXPECT_EQ(tachyon_bn254_plonk_proving_key_get_verifying_key(pkey),
            reinterpret_cast<const tachyon_bn254_plonk_verifying_key*>(
                &cpp_pkey.verifying_key()));
}

}  // namespace tachyon::zk::plonk
