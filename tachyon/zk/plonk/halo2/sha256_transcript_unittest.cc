// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/halo2/sha256_transcript.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"

namespace tachyon::zk::halo2 {

namespace {

using namespace math::bn254;

class Sha256TranscriptTest : public testing::Test {
 public:
  static void SetUpTestSuite() { G1Curve::Init(); }
};

}  // namespace

TEST_F(Sha256TranscriptTest, WritePoint) {
  base::Uint8VectorBuffer write_buf;
  Sha256Writer<G1Curve> writer(std::move(write_buf));
  G1AffinePoint expected = G1AffinePoint::Random();
  ASSERT_TRUE(writer.WriteToProof</*NeedToWriteToTranscript=*/true>(expected));

  base::Buffer read_buf(writer.buffer().buffer(), writer.buffer().buffer_len());
  Sha256Reader<G1Curve> reader(std::move(read_buf));
  G1AffinePoint actual;
  ASSERT_TRUE(reader.ReadFromProof</*NeedToWriteToTranscript=*/true>(&actual));

  EXPECT_EQ(expected, actual);
}

TEST_F(Sha256TranscriptTest, WriteScalar) {
  base::Uint8VectorBuffer write_buf;
  Sha256Writer<G1Curve> writer(std::move(write_buf));
  Fr expected = Fr::Random();
  ASSERT_TRUE(writer.WriteToProof</*NeedToWriteToTranscript=*/true>(expected));

  base::Buffer read_buf(writer.buffer().buffer(), writer.buffer().buffer_len());
  Sha256Reader<G1Curve> reader(std::move(read_buf));
  Fr actual;
  ASSERT_TRUE(reader.ReadFromProof</*NeedToWriteToTranscript=*/true>(&actual));

  EXPECT_EQ(expected, actual);
}

TEST_F(Sha256TranscriptTest, SqueezeChallenge) {
  base::Uint8VectorBuffer write_buf;
  Sha256Writer<G1Curve> writer(std::move(write_buf));
  G1AffinePoint generator = G1AffinePoint::Generator();
  ASSERT_TRUE(writer.WriteToProof</*NeedToWriteToTranscript=*/true>(generator));

  std::vector<uint8_t> expected_bytes = {144, 70,  170, 43,  125, 191, 116, 100,
                                         115, 242, 37,  247, 43,  227, 23,  192,
                                         153, 176, 105, 131, 142, 165, 91,  3,
                                         218, 85,  31,  89,  176, 94,  171, 5};
  Fr expected = Fr::FromBigInt(math::BigInt<4>::FromBytesLE(expected_bytes));

  Fr actual = writer.SqueezeChallenge();

  EXPECT_EQ(expected, actual);
}

}  // namespace tachyon::zk::halo2
