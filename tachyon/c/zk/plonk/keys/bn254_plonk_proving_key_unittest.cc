#include "tachyon/c/zk/plonk/keys/bn254_plonk_proving_key.h"

#include "gtest/gtest.h"

#include "tachyon/c/zk/plonk/halo2/bn254_halo2_ls.h"
#include "tachyon/c/zk/plonk/halo2/constants.h"
#include "tachyon/c/zk/plonk/keys/bn254_plonk_verifying_key_type_traits.h"
#include "tachyon/c/zk/plonk/keys/proving_key_impl.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::zk::plonk {

namespace {

class Bn254PlonkProvingKeyTest : public math::FiniteFieldTest<math::bn254::Fr> {
};

}  // namespace

TEST_F(Bn254PlonkProvingKeyTest, GetVerifyingKey) {
  using PkeyImpl =
      c::zk::plonk::ProvingKeyImpl<c::zk::plonk::halo2::bn254::Halo2LS>;
  PkeyImpl cpp_pkey;

  tachyon_bn254_plonk_proving_key pkey;
  pkey.ls_type = TACHYON_HALO2_HALO2_LS;
  pkey.extra = &cpp_pkey;
  EXPECT_EQ(tachyon_bn254_plonk_proving_key_get_verifying_key(&pkey),
            c::base::c_cast(&cpp_pkey.verifying_key()));
}

}  // namespace tachyon::zk::plonk
