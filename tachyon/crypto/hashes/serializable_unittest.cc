#include "gmock/gmock.h"
#include "gtest/gtest.h"
//clang-format off
#include "tachyon/base/logging.h"
#include "tachyon/crypto/hashes/bytes_serializable.h"
#include "tachyon/crypto/hashes/growable_buffer.h"
#include "tachyon/crypto/hashes/prime_field_serializable.h"
#include "tachyon/math/finite_fields/test/gf7.h"
// clang-format on

namespace tachyon::crypto {

namespace {

template <typename PrimeFieldType>
class SerializableTest : public testing::Test {
 public:
  static void SetUpTestSuite() {
#if defined(TACHYON_GMP_BACKEND)
    if constexpr (std::is_same_v<PrimeFieldType, math::GF7Gmp>) {
      PrimeFieldType::Init();
    }
#endif  // defined(TACHYON_GMP_BACKEND)
  }
};

}  // namespace

using PrimeFieldTypes = testing::Types<math::GF7Config>;
TYPED_TEST_SUITE(SerializableTest, PrimeFieldTypes);

TYPED_TEST(SerializableTest, SerializeInt) {
  constexpr int kValue = 12345;

  GrowableBuffer buf;
  buf.set_buffer_offset(0);
  ASSERT_TRUE(SerializeToBytes(kValue, &buf));
  EXPECT_EQ(buf.buffer_offset(), sizeof(int));
  EXPECT_EQ(*reinterpret_cast<const int*>(buf.buffer()), kValue);
}

TYPED_TEST(SerializableTest, SerializeFloat) {
  constexpr float kValue = 3.14f;

  GrowableBuffer buf;
  buf.set_buffer_offset(0);
  ASSERT_TRUE(SerializeToBytes(kValue, &buf));
  EXPECT_EQ(buf.buffer_offset(), sizeof(float));
  EXPECT_EQ(*reinterpret_cast<const float*>(buf.buffer()), kValue);
}

TYPED_TEST(SerializableTest, SerializeVector) {
  std::vector<int> values = {1, 2, 3, 4, 5};

  GrowableBuffer buf;
  buf.set_buffer_offset(0);
  ASSERT_TRUE(BatchSerializeToBytes(values, &buf));
  EXPECT_EQ(buf.buffer_offset(),
            BytesSerializable<std::vector<int>>::GetSize(values));
  const int* data_ptr = reinterpret_cast<const int*>(buf.buffer());
  for (int i = 0; i < values.size(); ++i) {
    EXPECT_EQ(data_ptr[i], values[i]);
  }
}

TYPED_TEST(SerializableTest, SerializeSingleValueToField) {
  using F = TypeParam;
  constexpr int64_t kValue = 5;
  std::vector<math::PrimeField<F>> fields;
  ASSERT_TRUE(SerializeToFieldElements(kValue, &fields));
  EXPECT_EQ(fields.size(), 1);
  EXPECT_EQ(fields[0].ToString(), "5");
}

TYPED_TEST(SerializableTest, SerializeBatchToField) {
  using F = TypeParam;
  std::vector<int64_t> values = {1, 2, 3, 4, 5};
  std::vector<math::PrimeField<F>> fields;
  ASSERT_TRUE(
      PrimeFieldSerializable<int64_t>::BatchToPrimeField(values, &fields));

  EXPECT_EQ(fields.size(), values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(fields[i].ToString(), std::to_string(values[i]));
  }
}

TYPED_TEST(SerializableTest, SerializationFailureDueToModulus) {
  using F = TypeParam;
  int64_t value = F::kModulus[0];
  std::vector<math::PrimeField<F>> fields;
  ASSERT_FALSE(SerializeToFieldElements(value, &fields));
  EXPECT_TRUE(fields.empty());
}

}  // namespace tachyon::crypto
