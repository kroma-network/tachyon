#include "circomlib/json/groth16_proof.h"

#include <string>

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/bn254.h"

namespace tachyon::circom {

namespace {

using Curve = math::bn254::BN254Curve;

class Groth16ProofTest : public testing::Test {
 public:
  static void SetUpTestSuite() { Curve::Init(); }
};

}  // namespace

TEST_F(Groth16ProofTest, Write) {
  zk::r1cs::groth16::Proof<Curve> proof(math::bn254::G1AffinePoint::Random(),
                                        math::bn254::G2AffinePoint::Random(),
                                        math::bn254::G1AffinePoint::Random());

  rapidjson::Document document = ConvertToJson(proof);
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  document.Accept(writer);
  {
    auto it = document.GetObject().FindMember("pi_a");
    ASSERT_NE(it, document.MemberEnd());
    auto it2 = it->value.GetArray().Begin();
    for (size_t i = 0; i < 3; ++i) {
      std::string expected;
      if (i == 0) {
        expected = proof.a().x().ToString();
      } else if (i == 1) {
        expected = proof.a().y().ToString();
      } else {
        expected = "1";
      }
      EXPECT_EQ(*(it2++), expected);
    }
    ASSERT_EQ(it2, it->value.GetArray().End());
  }
  {
    auto it = document.GetObject().FindMember("pi_b");
    ASSERT_NE(it, document.MemberEnd());
    auto it2 = it->value.GetArray().Begin();
    for (size_t i = 0; i < 3; ++i) {
      std::string expected;
      if (i == 0) {
        auto it3 = it2->GetArray().Begin();
        for (size_t j = 0; j < 2; ++j) {
          if (j == 0) {
            expected = proof.b().x().c0().ToString();
          } else {
            expected = proof.b().x().c1().ToString();
          }
          EXPECT_EQ(*(it3++), expected);
        }
        ASSERT_EQ(it3, it2->GetArray().End());
      } else if (i == 1) {
        auto it3 = it2->GetArray().Begin();
        for (size_t j = 0; j < 2; ++j) {
          if (j == 0) {
            expected = proof.b().y().c0().ToString();
          } else {
            expected = proof.b().y().c1().ToString();
          }
          EXPECT_EQ(*(it3++), expected);
        }
        ASSERT_EQ(it3, it2->GetArray().End());
      } else {
        auto it3 = it2->GetArray().Begin();
        for (size_t j = 0; j < 2; ++j) {
          if (j == 0) {
            expected = "1";
          } else {
            expected = "0";
          }
          EXPECT_EQ(*(it3++), expected);
        }
        ASSERT_EQ(it3, it2->GetArray().End());
      }
      ++it2;
    }
    ASSERT_EQ(it2, it->value.GetArray().End());
  }
  {
    auto it = document.GetObject().FindMember("pi_c");
    auto it2 = it->value.GetArray().Begin();
    for (size_t i = 0; i < 3; ++i) {
      std::string expected;
      if (i == 0) {
        expected = proof.c().x().ToString();
      } else if (i == 1) {
        expected = proof.c().y().ToString();
      } else {
        expected = "1";
      }
      EXPECT_EQ(*(it2++), expected);
    }
    ASSERT_EQ(it2, it->value.GetArray().End());
  }
  {
    auto it = document.GetObject().FindMember("protocol");
    ASSERT_NE(it, document.MemberEnd());
    EXPECT_EQ(it->value.GetString(), "groth16");
  }
  {
    auto it = document.GetObject().FindMember("curve");
    ASSERT_NE(it, document.MemberEnd());
    EXPECT_EQ(it->value.GetString(), "bn128");
  }
}

}  // namespace tachyon::circom
