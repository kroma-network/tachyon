#include "circomlib/wtns/wtns.h"

#include <vector>

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::circom {

using F = math::bn254::Fr;

namespace {

class WtnsTest : public math::FiniteFieldTest<F> {};

}  // namespace

TEST(WtnsTest, Parse) {
  // Generated with { "in": ["3", "4", "5"] }
  std::unique_ptr<Wtns<F>> wtns =
      ParseWtns<F>(base::FilePath("circomlib/wtns/multiplier_3.wtns"));
  ASSERT_TRUE(wtns);
  ASSERT_EQ(wtns->GetVersion(), 2);

  std::array<uint8_t, 32> bytes = math::bn254::FrConfig::kModulus.ToBytesLE();
  v2::WtnsHeaderSection expected_header = {
      Modulus{std::vector<uint8_t>(bytes.begin(), bytes.end())},
      6,
  };
  EXPECT_EQ(wtns->ToV2()->header, expected_header);

  std::vector<F> expected_data{
      F(1), F(60), F(3), F(4), F(5), F(12),
  };
  v2::WtnsDataSection<F> expected_data_section = {
      absl::MakeSpan(expected_data)};
  EXPECT_EQ(wtns->ToV2()->data, expected_data_section);
}

}  // namespace tachyon::circom
