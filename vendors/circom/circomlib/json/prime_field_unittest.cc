#include "circomlib/json/prime_field.h"

#include <vector>

#include "gtest/gtest.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::circom {

namespace {

class PrimeFieldTest : public math::FiniteFieldTest<math::bn254::Fr> {};

}  // namespace

TEST_F(PrimeFieldTest, Write) {
  std::vector<math::bn254::Fr> prime_fields =
      base::CreateVector(8, []() { return math::bn254::Fr::Random(); });

  rapidjson::Document document =
      ConvertToJson(absl::MakeConstSpan(prime_fields));
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  document.Accept(writer);
  auto it = document.Begin();
  auto it2 = prime_fields.begin();
  for (; it != document.End(); ++it, ++it2) {
    EXPECT_EQ(it->GetString(), it2->ToString());
  }
}

}  // namespace tachyon::circom
