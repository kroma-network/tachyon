#include "tachyon/zk/plonk/keys/verifying_key.h"

#include "gtest/gtest.h"

#include "tachyon/crypto/commitments/kzg/kzg_commitment_scheme.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g2.h"

namespace tachyon::zk {

namespace {

class VerifyingKeyTest : public testing::Test {
 public:
  constexpr static size_t kMaxDegree = 7;

  using PCS =
      crypto::KZGCommitmentScheme<math::bn254::G1AffinePoint,
                                  math::bn254::G2AffinePoint, kMaxDegree,
                                  math::bn254::G1AffinePoint>;

  static void SetUpTestSuite() { math::bn254::G1Curve::Init(); }
};

}  // namespace

// TODO(chokobole): Implement test codes.
TEST_F(VerifyingKeyTest, Generate) { VerifyingKey<PCS> verifying_key; }

}  // namespace tachyon::zk
