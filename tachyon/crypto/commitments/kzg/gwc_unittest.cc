#include "tachyon/crypto/commitments/kzg/gwc.h"

#include "tachyon/crypto/commitments/kzg/kzg_family_test.h"

namespace tachyon::crypto {
namespace {

using PCS =
    GWC<math::bn254::BN254Curve, kMaxDegree, math::bn254::G1AffinePoint>;

class GWCTest : public KZGFamilyTest<PCS> {};

}  // namespace

TEST_F(GWCTest, CreateAndVerifyProof) { this->CreateAndVerifyProof(); }

TEST_F(GWCTest, Copyable) { this->Copyable(); }

}  // namespace tachyon::crypto
