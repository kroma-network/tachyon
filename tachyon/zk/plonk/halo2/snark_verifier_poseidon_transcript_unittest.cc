#include "tachyon/zk/plonk/halo2/snark_verifier_poseidon_transcript.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"

namespace tachyon::zk::plonk::halo2 {

namespace {

using namespace math::bn254;

class SnarkVerifierPoseidonTranscriptTest : public testing::Test {
 public:
  static void SetUpTestSuite() { G1Curve::Init(); }
};

}  // namespace

TEST_F(SnarkVerifierPoseidonTranscriptTest, WritePoint) {
  base::Uint8VectorBuffer write_buf;
  SnarkVerifierPoseidonWriter<G1AffinePoint> writer(std::move(write_buf));
  G1AffinePoint expected = G1AffinePoint::Random();
  ASSERT_TRUE(writer.WriteToProof(expected));

  base::Buffer read_buf(writer.buffer().buffer(), writer.buffer().buffer_len());
  SnarkVerifierPoseidonReader<G1AffinePoint> reader(std::move(read_buf));
  G1AffinePoint actual;
  ASSERT_TRUE(reader.ReadFromProof(&actual));

  EXPECT_EQ(expected, actual);
}

TEST_F(SnarkVerifierPoseidonTranscriptTest, WriteScalar) {
  base::Uint8VectorBuffer write_buf;
  SnarkVerifierPoseidonWriter<G1AffinePoint> writer(std::move(write_buf));
  Fr expected = Fr::Random();
  ASSERT_TRUE(writer.WriteToProof(expected));

  base::Buffer read_buf(writer.buffer().buffer(), writer.buffer().buffer_len());
  SnarkVerifierPoseidonReader<G1AffinePoint> reader(std::move(read_buf));
  Fr actual;
  ASSERT_TRUE(reader.ReadFromProof(&actual));

  EXPECT_EQ(expected, actual);
}

TEST_F(SnarkVerifierPoseidonTranscriptTest, SqueezeChallenge) {
  base::Uint8VectorBuffer write_buf;
  SnarkVerifierPoseidonWriter<G1AffinePoint> writer(std::move(write_buf));
  G1AffinePoint generator = G1AffinePoint::Generator();
  ASSERT_TRUE(writer.WriteToProof(generator));
  ASSERT_TRUE(writer.WriteToProof(generator));

  std::vector<uint8_t> expected_bytes = {78,  246, 205, 146, 54,  16,  105, 106,
                                         240, 24,  115, 146, 126, 203, 44,  166,
                                         34,  117, 244, 97,  33,  69,  158, 167,
                                         254, 239, 174, 66,  133, 142, 174, 27};
  Fr expected = Fr::FromBigInt(math::BigInt<4>::FromBytesLE(expected_bytes));

  Fr actual = writer.SqueezeChallenge();

  EXPECT_EQ(expected, actual);
}

}  // namespace tachyon::zk::plonk::halo2
