// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/halo2/blake2b_transcript.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"

namespace tachyon::zk::halo2 {

namespace {

using namespace math::bn254;

class Blake2bTranscriptTest : public testing::Test {
 public:
  static void SetUpTestSuite() { G1Curve::Init(); }
};

}  // namespace

TEST_F(Blake2bTranscriptTest, WritePoint) {
  base::Uint8VectorBuffer write_buf;
  Blake2bWriter<G1AffinePoint> writer(std::move(write_buf));
  G1AffinePoint expected = G1AffinePoint::Random();
  ASSERT_TRUE(writer.WriteToProof</*NeedToWriteToTranscript=*/true>(expected));

  base::Buffer read_buf(writer.buffer().buffer(), writer.buffer().buffer_len());
  Blake2bReader<G1AffinePoint> reader(std::move(read_buf));
  G1AffinePoint actual;
  ASSERT_TRUE(reader.ReadFromProof</*NeedToWriteToTranscript=*/true>(&actual));

  EXPECT_EQ(expected, actual);
}

TEST_F(Blake2bTranscriptTest, WriteScalar) {
  base::Uint8VectorBuffer write_buf;
  Blake2bWriter<G1AffinePoint> writer(std::move(write_buf));
  Fr expected = Fr::Random();
  ASSERT_TRUE(writer.WriteToProof</*NeedToWriteToTranscript=*/true>(expected));

  base::Buffer read_buf(writer.buffer().buffer(), writer.buffer().buffer_len());
  Blake2bReader<G1AffinePoint> reader(std::move(read_buf));
  Fr actual;
  ASSERT_TRUE(reader.ReadFromProof</*NeedToWriteToTranscript=*/true>(&actual));

  EXPECT_EQ(expected, actual);
}

TEST_F(Blake2bTranscriptTest, SqueezeChallenge) {
  base::Uint8VectorBuffer write_buf;
  Blake2bWriter<G1AffinePoint> writer(std::move(write_buf));
  G1AffinePoint generator = G1AffinePoint::Generator();
  ASSERT_TRUE(writer.WriteToProof</*NeedToWriteToTranscript=*/true>(generator));

  std::vector<uint8_t> expected_bytes = {57, 2,   118, 182, 16,  184, 59,  179,
                                         70, 176, 223, 71,  62,  168, 222, 171,
                                         85, 224, 83,  43,  148, 194, 132, 184,
                                         65, 25,  1,   208, 123, 166, 11,  12};
  Fr expected = Fr::FromBigInt(math::BigInt<4>::FromBytesLE(expected_bytes));

  Fr actual = writer.SqueezeChallenge();

  EXPECT_EQ(expected, actual);
}

}  // namespace tachyon::zk::halo2
