#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/buffer/vector_buffer.h"

namespace tachyon::base {

TEST(CopyableTest, BuiltInSerializableTest) {
#define TEST_BUILTIN_TYPES(type)                                   \
  EXPECT_TRUE(base::internal::IsBuiltinSerializable<type>::value); \
  EXPECT_FALSE(base::internal::IsNonBuiltinSerializable<type>::value)

#define TEST_NON_BUILTIN_TYPES(type)                                \
  EXPECT_FALSE(base::internal::IsBuiltinSerializable<type>::value); \
  EXPECT_TRUE(base::internal::IsNonBuiltinSerializable<type>::value)

  TEST_BUILTIN_TYPES(bool);
  TEST_BUILTIN_TYPES(char);
  TEST_BUILTIN_TYPES(uint16_t);
  TEST_BUILTIN_TYPES(uint32_t);
  TEST_BUILTIN_TYPES(uint64_t);
  TEST_BUILTIN_TYPES(int16_t);
  TEST_BUILTIN_TYPES(int32_t);
  TEST_BUILTIN_TYPES(int64_t);

  enum class Color {
    kRed,
    kBlue,
    kGreen,
  };
  TEST_BUILTIN_TYPES(Color);

  TEST_NON_BUILTIN_TYPES(std::string_view);
  TEST_NON_BUILTIN_TYPES(std::string);
  TEST_NON_BUILTIN_TYPES(uint64_t[4]);
  TEST_NON_BUILTIN_TYPES(std::vector<uint64_t>);

  using Array = std::array<uint64_t, 4>;
  TEST_NON_BUILTIN_TYPES(Array);

#undef TEST_NON_BUILTIN_TYPES
#undef TEST_BUILTIN_TYPES
}

TEST(BufferTest, Write) {
  constexpr char kCharValue = 'c';
  constexpr int kIntValue = 12345;
  constexpr bool kBooleanValue = true;
  const char* kCharPtrValue = "abc";
  std::string kStringValue = "def";
  uint64_t kIntBoundedArray[4] = {1, 2, 3, 4};
  std::vector<int> kIntVector = {5, 6, 7};
  std::array<int, 4> kIntArray = {8, 9, 10, 11};

  for (Endian endian : {Endian::kNative, Endian::kBig, Endian::kLittle}) {
    VectorBuffer write_buf;
    write_buf.set_endian(endian);
    ASSERT_TRUE(write_buf.Write(kCharValue));
    ASSERT_TRUE(write_buf.Write(kIntValue));
    ASSERT_TRUE(write_buf.Write(kBooleanValue));
    ASSERT_TRUE(write_buf.Write(kCharPtrValue));
    ASSERT_TRUE(write_buf.Write(kStringValue));
    ASSERT_TRUE(write_buf.Write(kIntBoundedArray));
    ASSERT_TRUE(write_buf.Write(kIntVector));
    ASSERT_TRUE(write_buf.Write(kIntArray));

    Buffer read_buf(write_buf.buffer(), write_buf.buffer_len());
    read_buf.set_endian(endian);
    char c;
    ASSERT_TRUE(read_buf.Read(&c));
    int i;
    ASSERT_TRUE(read_buf.Read(&i));
    bool b;
    ASSERT_TRUE(read_buf.Read(&b));
    std::string s;
    ASSERT_TRUE(read_buf.Read(&s));
    std::string s2;
    ASSERT_TRUE(read_buf.Read(&s2));
    uint64_t iba[4];
    ASSERT_TRUE(read_buf.Read(iba));
    std::vector<int> iv;
    ASSERT_TRUE(read_buf.Read(&iv));
    std::array<int, 4> ia;
    ASSERT_TRUE(read_buf.Read(&ia));
    ASSERT_TRUE(read_buf.Done());
    EXPECT_EQ(c, kCharValue);
    EXPECT_EQ(i, kIntValue);
    EXPECT_EQ(b, kBooleanValue);
    EXPECT_EQ(s, kCharPtrValue);
    EXPECT_EQ(s2, kStringValue);
    EXPECT_THAT(iba, testing::ElementsAreArray(iba));
    EXPECT_EQ(iv, kIntVector);
    EXPECT_EQ(ia, kIntArray);
  }
}

TEST(BufferTest, WriteMany) {
  constexpr char kCharValue = 'c';
  constexpr int kIntValue = 12345;
  constexpr bool kBooleanValue = true;
  const char* kCharPtrValue = "abc";
  std::string kStringValue = "def";
  uint64_t kIntBoundedArray[4] = {1, 2, 3, 4};
  std::vector<int> kIntVector = {5, 6, 7};
  std::array<int, 4> kIntArray = {8, 9, 10, 11};

  for (Endian endian : {Endian::kNative, Endian::kBig, Endian::kLittle}) {
    VectorBuffer write_buf;
    write_buf.set_endian(endian);
    ASSERT_TRUE(write_buf.WriteMany(kCharValue, kIntValue, kBooleanValue,
                                    kCharPtrValue, kStringValue,
                                    kIntBoundedArray, kIntVector, kIntArray));

    Buffer read_buf(write_buf.buffer(), write_buf.buffer_len());
    read_buf.set_endian(endian);
    char c;
    int i;
    bool b;
    std::string s;
    std::string s2;
    uint64_t iba[4];
    std::vector<int> iv;
    std::array<int, 4> ia;
    ASSERT_TRUE(read_buf.ReadMany(&c, &i, &b, &s, &s2, iba, &iv, &ia));
    EXPECT_EQ(c, kCharValue);
    EXPECT_EQ(i, kIntValue);
    EXPECT_EQ(b, kBooleanValue);
    EXPECT_EQ(s, kCharPtrValue);
    EXPECT_EQ(s2, kStringValue);
    EXPECT_THAT(iba, testing::ElementsAreArray(iba));
    EXPECT_EQ(iv, kIntVector);
    EXPECT_EQ(ia, kIntArray);
    ASSERT_TRUE(read_buf.Done());
  }
}

}  // namespace tachyon::base
