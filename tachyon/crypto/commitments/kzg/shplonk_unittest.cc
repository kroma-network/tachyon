#include "tachyon/crypto/commitments/kzg/shplonk.h"

#include "tachyon/crypto/commitments/kzg/kzg_family_test.h"

namespace tachyon::crypto {
namespace {

using PCS =
    SHPlonk<math::bn254::BN254Curve, kMaxDegree, math::bn254::G1AffinePoint>;

class SHPlonkTest : public KZGFamilyTest<PCS> {};

}  // namespace

TEST_F(SHPlonkTest, CreateAndVerifyProof) { this->CreateAndVerifyProof(); }

TEST_F(SHPlonkTest, Copyable) { this->Copyable(); }

}  // namespace tachyon::crypto
