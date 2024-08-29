#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_opened_values.h"

#include <vector>

#include "gtest/gtest.h"

#include "tachyon/base/auto_reset.h"
#include "tachyon/base/buffer/read_only_buffer.h"
#include "tachyon/base/random.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_opened_values_type_traits.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::math {

namespace {

using ExtF = BabyBear4;
using OpenedValues = std::vector<std::vector<std::vector<std::vector<ExtF>>>>;

class OpenedValuesTest : public FiniteFieldTest<ExtF> {};

size_t GetRandom() { return base::Uniform(base::Range<size_t>(1, 3)); }

}  // namespace

TEST_F(OpenedValuesTest, Serialize) {
  OpenedValues opened_values = base::CreateVector(GetRandom(), []() {
    return base::CreateVector(GetRandom(), []() {
      return base::CreateVector(GetRandom(), []() {
        return base::CreateVector(GetRandom(), []() { return ExtF::Random(); });
      });
    });
  });
  size_t data_len;
  tachyon_sp1_baby_bear_poseidon2_opened_values_serialize(
      c::base::c_cast(&opened_values), nullptr, &data_len);
  ASSERT_NE(data_len, size_t{0});

  std::vector<uint8_t> data(data_len);
  tachyon_sp1_baby_bear_poseidon2_opened_values_serialize(
      c::base::c_cast(&opened_values), const_cast<uint8_t*>(data.data()),
      &data_len);
  ASSERT_EQ(data.size(), data_len);

  base::ReadOnlyBuffer buffer(data.data(), data_len);
  OpenedValues opened_values_deser;
  base::AutoReset<bool> auto_reset(
      &base::Copyable<BabyBear>::s_is_in_montgomery, true);
  ASSERT_TRUE(buffer.Read(&opened_values_deser));
  EXPECT_EQ(opened_values, opened_values_deser);
}

}  // namespace tachyon::math
