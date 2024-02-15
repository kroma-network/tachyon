#include "gtest/gtest.h"

#include "tachyon/base/buffer/buffer.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fq12.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::math {

namespace {

class Fp12Test : public FiniteFieldTest<bn254::Fq12> {};

}  // namespace

TEST_F(Fp12Test, TypeTest) {
  EXPECT_TRUE((std::is_same_v<bn254::Fq12::BaseField, bn254::Fq6>));
  EXPECT_TRUE((std::is_same_v<bn254::Fq12::BasePrimeField, bn254::Fq>));
}

TEST_F(Fp12Test, Copyable) {
  using F = bn254::Fq12;

  const F expected = F::Random();

  std::vector<uint8_t> vec;
  vec.resize(base::EstimateSize(expected));
  base::Buffer write_buf(vec.data(), vec.size());
  ASSERT_TRUE(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  F value;
  ASSERT_TRUE(write_buf.Read(&value));
  EXPECT_EQ(expected, value);
}

TEST_F(Fp12Test, JsonValueConverter) {
  using F = bn254::Fq12;

  F expected = F::Random();
  std::string json = base::WriteToJson(expected);

  F value;
  std::string error;
  ASSERT_TRUE(base::ParseJson(json, &value, &error));
  ASSERT_TRUE(error.empty());
  EXPECT_EQ(value, expected);
}

}  // namespace tachyon::math
