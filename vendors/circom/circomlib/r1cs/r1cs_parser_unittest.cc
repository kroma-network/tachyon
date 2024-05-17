#include "circomlib/r1cs/r1cs_parser.h"

#include <vector>

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::circom {

using F = math::bn254::Fr;

namespace {

class R1CSParserTest : public math::FiniteFieldTest<F> {};

}  // namespace

TEST_F(R1CSParserTest, Parse) {
  R1CSParser<F> parser;
  std::unique_ptr<R1CS<F>> r1cs =
      parser.Parse(base::FilePath("examples/multiplier_3.r1cs"));
  ASSERT_TRUE(r1cs);
  ASSERT_EQ(r1cs->GetVersion(), 1);

  std::array<uint8_t, 32> bytes = math::bn254::FrConfig::kModulus.ToBytesLE();
  v1::R1CSHeaderSection expected_header = {
      PrimeField{std::vector<uint8_t>(bytes.begin(), bytes.end())},
      6,
      1,
      0,
      3,
      11,
      2,
  };
  EXPECT_EQ(r1cs->ToV1()->header, expected_header);

  F one = F::One();
  F neg_one(math::bn254::FrConfig::kModulus - math::BigInt<4>::One());
  // -ω₂ * ω₃ = -ω₅
  // -ω₅ * ω₄ = -ω₁
  v1::R1CSConstraintsSection<F> expected_constraints = {{{
                                                             {{{2, neg_one}}},
                                                             {{{3, one}}},
                                                             {{{5, neg_one}}},
                                                         },
                                                         {
                                                             {{{5, neg_one}}},
                                                             {{{4, one}}},
                                                             {{{1, neg_one}}},
                                                         }}};
  EXPECT_EQ(r1cs->ToV1()->constraints, expected_constraints);
  v1::R1CSWireId2LabelIdMapSection expected_wire_id_to_label_id_map = {{
      0,
      1,
      2,
      3,
      4,
      5,
  }};
  EXPECT_EQ(r1cs->ToV1()->wire_id_to_label_id_map,
            expected_wire_id_to_label_id_map);
}

}  // namespace tachyon::circom
