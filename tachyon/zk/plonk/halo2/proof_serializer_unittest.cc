// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/halo2/proof_serializer.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"

namespace tachyon::zk::plonk::halo2 {

namespace {

using namespace math::bn254;

class ProofSerializerTest : public testing::Test {
 public:
  static void SetUpTestSuite() { G1Curve::Init(); }
};

}  // namespace

TEST_F(ProofSerializerTest, SerializeScalar) {
  struct {
    std::string_view hex;
    std::vector<uint8_t> proof;
  } tests[] = {
      {"0x2482c9ce1f365ed93c2afe1df9c673b0ba65278badd4d150f3b848cdd3d0cec8",
       {200, 206, 208, 211, 205, 72,  184, 243, 80,  209, 212,
        173, 139, 39,  101, 186, 176, 115, 198, 249, 29,  254,
        42,  60,  217, 94,  54,  31,  206, 201, 130, 36}},
  };

  for (const auto& test : tests) {
    std::vector<uint8_t> buffer;
    buffer.resize(test.proof.size());
    base::Buffer write_buf(buffer.data(), buffer.size());
    Fr expected = Fr::FromHexString(test.hex);
    ASSERT_TRUE(ProofSerializer<Fr>::WriteToProof(expected, write_buf));
    EXPECT_THAT(buffer, testing::ElementsAreArray(test.proof));

    write_buf.set_buffer_offset(0);
    Fr actual;
    ASSERT_TRUE(ProofSerializer<Fr>::ReadFromProof(write_buf, &actual));
    EXPECT_EQ(actual, expected);
  }
}

TEST_F(ProofSerializerTest, SerializeProof) {
  struct {
    std::array<std::string_view, 2> hex;
    std::vector<uint8_t> proof;
  } tests[] = {
      // even point
      {{
           "0x233bd4dc42ffd123f6d041dca2117acea5f6a201b4612a81e7081cad001df470",
           "0x14ecc49a7d74ee9059862ca5237c72f22dc6c39b64ec3e7c4ec314187577ee56",
       },
       {112, 244, 29,  0,   173, 28,  8,   231, 129, 42,  97,
        180, 1,   162, 246, 165, 206, 122, 17,  162, 220, 65,
        208, 246, 35,  209, 255, 66,  220, 212, 59,  35}},
      // odd point
      {{
           "0x1ec72fa9df2846c267ad6bc77e438c0d8c0c9bba978be3095cc48b0334299dbb",
           "0x2c1b5dfdca4dfc40a864355fead42fb3656a8a3304ad11b1dee1a4b924ac5a03",
       },
       {187, 157, 41,  52, 3,   139, 196, 92, 9,   227, 139,
        151, 186, 155, 12, 140, 13,  140, 67, 126, 199, 107,
        173, 103, 194, 70, 40,  223, 169, 47, 199, 158}},
  };

  for (const auto& test : tests) {
    std::vector<uint8_t> buffer;
    buffer.resize(test.proof.size());
    base::Buffer write_buf(buffer.data(), buffer.size());
    Fq x = Fq::FromHexString(test.hex[0]);
    Fq y = Fq::FromHexString(test.hex[1]);
    G1AffinePoint expected(x, y);
    ASSERT_TRUE(
        ProofSerializer<G1AffinePoint>::WriteToProof(expected, write_buf));
    EXPECT_THAT(buffer, testing::ElementsAreArray(test.proof));

    write_buf.set_buffer_offset(0);
    G1AffinePoint actual;
    ASSERT_TRUE(
        ProofSerializer<G1AffinePoint>::ReadFromProof(write_buf, &actual));
    EXPECT_EQ(actual, expected);
  }
}

}  // namespace tachyon::zk::plonk::halo2
