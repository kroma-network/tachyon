// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/halo2/poseidon_transcript.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"

namespace tachyon::zk::halo2 {

namespace {

using namespace math::bn254;

class PoseidonTranscriptTest : public testing::Test {
 public:
  static void SetUpTestSuite() { G1Curve::Init(); }
};

}  // namespace

TEST_F(PoseidonTranscriptTest, WritePoint) {
  base::Uint8VectorBuffer write_buf;
  PoseidonWriter<G1AffinePoint> writer(std::move(write_buf));
  G1AffinePoint expected = G1AffinePoint::Random();
  ASSERT_TRUE(writer.WriteToProof</*NeedToWriteToTranscript=*/true>(expected));

  base::Buffer read_buf(writer.buffer().buffer(), writer.buffer().buffer_len());
  PoseidonReader<G1AffinePoint> reader(std::move(read_buf));
  G1AffinePoint actual;
  ASSERT_TRUE(reader.ReadFromProof</*NeedToWriteToTranscript=*/true>(&actual));

  EXPECT_EQ(expected, actual);
}

TEST_F(PoseidonTranscriptTest, WriteScalar) {
  base::Uint8VectorBuffer write_buf;
  PoseidonWriter<G1AffinePoint> writer(std::move(write_buf));
  Fr expected = Fr::Random();
  ASSERT_TRUE(writer.WriteToProof</*NeedToWriteToTranscript=*/true>(expected));

  base::Buffer read_buf(writer.buffer().buffer(), writer.buffer().buffer_len());
  PoseidonReader<G1AffinePoint> reader(std::move(read_buf));
  Fr actual;
  ASSERT_TRUE(reader.ReadFromProof</*NeedToWriteToTranscript=*/true>(&actual));

  EXPECT_EQ(expected, actual);
}

TEST_F(PoseidonTranscriptTest, SqueezeChallenge) {
  base::Uint8VectorBuffer write_buf;
  PoseidonWriter<G1AffinePoint> writer(std::move(write_buf));
  G1AffinePoint generator = G1AffinePoint::Generator();
  ASSERT_TRUE(writer.WriteToProof</*NeedToWriteToTranscript=*/true>(generator));

  std::vector<uint8_t> expected_bytes = {25,  86,  205, 219, 59,  135, 187, 231,
                                         192, 54,  23,  138, 114, 176, 9,   157,
                                         1,   97,  110, 174, 67,  9,   89,  85,
                                         126, 129, 216, 121, 53,  99,  227, 26};
  Fr expected = Fr::FromBigInt(math::BigInt<4>::FromBytesLE(expected_bytes));

  Fr actual = writer.SqueezeChallenge();

  EXPECT_EQ(expected, actual);
}

}  // namespace tachyon::zk::halo2
