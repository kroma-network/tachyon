#include "tachyon/zk/plonk/keys/proving_key.h"

#include "gtest/gtest.h"

#include "tachyon/crypto/commitments/kzg/kzg_commitment_scheme.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g2.h"

namespace tachyon::zk {

namespace {

class ProvingKeyTest : public testing::Test {
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
TEST_F(ProvingKeyTest, Generate) { ProvingKey<PCS> proving_key; }

}  // namespace tachyon::zk
