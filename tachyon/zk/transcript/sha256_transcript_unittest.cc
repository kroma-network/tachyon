// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/transcript/sha256_transcript.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"

namespace tachyon::zk {

namespace {

class Sha256TranscriptTest : public testing::Test {
 public:
  static void SetUpTestSuite() { math::bn254::G1Curve::Init(); }
};

}  // namespace

TEST_F(Sha256TranscriptTest, WritePoint) {
  using AffinePoint = math::bn254::G1AffinePoint;

  base::VectorBuffer write_buf;
  Sha256Writer<AffinePoint> writer(std::move(write_buf));
  AffinePoint expected = AffinePoint::Random();
  ASSERT_TRUE(writer.WriteToProof(expected));

  base::Buffer read_buf(writer.buffer().buffer(), writer.buffer().buffer_len());
  Sha256Reader<AffinePoint> reader(std::move(read_buf));
  AffinePoint actual;
  ASSERT_TRUE(reader.ReadPoint(&actual));

  EXPECT_EQ(expected, actual);
}

TEST_F(Sha256TranscriptTest, WriteScalar) {
  using AffinePoint = math::bn254::G1AffinePoint;
  using ScalarField = AffinePoint::ScalarField;

  base::VectorBuffer write_buf;
  Sha256Writer<AffinePoint> writer(std::move(write_buf));
  ScalarField expected = ScalarField::Random();
  ASSERT_TRUE(writer.WriteToProof(expected));

  base::Buffer read_buf(writer.buffer().buffer(), writer.buffer().buffer_len());
  Sha256Reader<AffinePoint> reader(std::move(read_buf));
  ScalarField actual;
  ASSERT_TRUE(reader.ReadScalar(&actual));

  EXPECT_EQ(expected, actual);
}

TEST_F(Sha256TranscriptTest, SqueezeChallenge) {
  using AffinePoint = math::bn254::G1AffinePoint;
  using ScalarField = AffinePoint::ScalarField;

  base::VectorBuffer write_buf;
  Sha256Writer<AffinePoint> writer(std::move(write_buf));
  AffinePoint generator = AffinePoint::Generator();
  ASSERT_TRUE(writer.WriteToProof(generator));

  std::vector<uint8_t> expected_bytes = {144, 70,  170, 43,  125, 191, 116, 100,
                                         115, 242, 37,  247, 43,  227, 23,  192,
                                         153, 176, 105, 131, 142, 165, 91,  3,
                                         218, 85,  31,  89,  176, 94,  171, 5};
  ScalarField expected =
      ScalarField::FromBigInt(math::BigInt<4>::FromBytesLE(expected_bytes));

  ScalarField actual = writer.SqueezeChallenge().ChallengeAsScalar();

  EXPECT_EQ(expected, actual);
}

}  // namespace tachyon::zk
