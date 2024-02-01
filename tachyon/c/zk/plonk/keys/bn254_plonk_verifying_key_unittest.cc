#include "tachyon/c/zk/plonk/keys/bn254_plonk_verifying_key.h"

#include "gtest/gtest.h"

#include "tachyon/c/math/elliptic_curves/bn/bn254/fr_prime_field_traits.h"
#include "tachyon/cc/math/finite_fields/prime_field_conversions.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/zk/plonk/keys/verifying_key.h"

namespace tachyon::zk {

namespace {

class Bn254PlonkVerifyingKeyTest : public testing::Test {
 public:
  static void SetUpTestSuite() { math::bn254::Fr::Init(); }
};

}  // namespace

TEST_F(Bn254PlonkVerifyingKeyTest, GetTranscriptRepr) {
  VerifyingKey<math::bn254::Fr, math::bn254::G1AffinePoint> cpp_vkey;
  math::bn254::Fr cpp_transcript_repr = math::bn254::Fr::Random();
  cpp_vkey.SetTranscriptReprForTesting(cpp_transcript_repr);

  tachyon_bn254_plonk_verifying_key* vkey =
      reinterpret_cast<tachyon_bn254_plonk_verifying_key*>(&cpp_vkey);
  tachyon_bn254_fr transcript_repr =
      tachyon_bn254_plonk_verifying_key_get_transcript_repr(vkey);
  EXPECT_EQ(cc::math::ToPrimeField(transcript_repr), cpp_transcript_repr);
}

}  // namespace tachyon::zk
