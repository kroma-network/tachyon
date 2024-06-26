#include "tachyon/c/zk/plonk/keys/bn254_plonk_verifying_key.h"

#include "gtest/gtest.h"

#include "tachyon/c/math/elliptic_curves/bn/bn254/fr_type_traits.h"
#include "tachyon/c/zk/plonk/keys/bn254_plonk_verifying_key_type_traits.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::zk::plonk {

namespace {

class Bn254PlonkVerifyingKeyTest
    : public math::FiniteFieldTest<math::bn254::Fr> {};

}  // namespace

TEST_F(Bn254PlonkVerifyingKeyTest, GetTranscriptRepr) {
  VerifyingKey<math::bn254::Fr, math::bn254::G1AffinePoint> cpp_vkey;
  math::bn254::Fr cpp_transcript_repr = math::bn254::Fr::Random();
  cpp_vkey.SetTranscriptReprForTesting(cpp_transcript_repr);

  tachyon_bn254_plonk_verifying_key* vkey = c::base::c_cast(&cpp_vkey);
  tachyon_bn254_fr transcript_repr =
      tachyon_bn254_plonk_verifying_key_get_transcript_repr(vkey);
  EXPECT_EQ(c::base::native_cast(transcript_repr), cpp_transcript_repr);
}

}  // namespace tachyon::zk::plonk
