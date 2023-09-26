#include "gtest/gtest.h"

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/buffer/vector_buffer.h"

namespace tachyon::base {

TEST(BufferTest, Write) {
  constexpr char kCharValue = 'c';
  constexpr int kIntValue = 12345;
  constexpr bool kBooleanValue = true;
  const char* kCharPtrValue = "abc";
  std::string kStringValue = "def";
  std::vector<int> kIntVector = {1, 2, 3};
  std::array<int, 4> kIntArray = {4, 5, 6, 7};

  for (Endian endian : {Endian::kNative, Endian::kBig, Endian::kLittle}) {
    VectorBuffer write_buf;
    write_buf.set_endian(endian);
    ASSERT_TRUE(write_buf.Write(kCharValue));
    ASSERT_TRUE(write_buf.Write(kIntValue));
    ASSERT_TRUE(write_buf.Write(kBooleanValue));
    ASSERT_TRUE(write_buf.Write(kCharPtrValue));
    ASSERT_TRUE(write_buf.Write(kStringValue));
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
  std::vector<int> kIntVector = {1, 2, 3};
  std::array<int, 4> kIntArray = {4, 5, 6, 7};

  for (Endian endian : {Endian::kNative, Endian::kBig, Endian::kLittle}) {
    VectorBuffer write_buf;
    write_buf.set_endian(endian);
    ASSERT_TRUE(write_buf.WriteMany(kCharValue, kIntValue, kBooleanValue,
                                    kCharPtrValue, kStringValue, kIntVector,
                                    kIntArray));

    Buffer read_buf(write_buf.buffer(), write_buf.buffer_len());
    read_buf.set_endian(endian);
    char c;
    int i;
    bool b;
    std::string s;
    std::string s2;
    std::vector<int> iv;
    std::array<int, 4> ia;
    ASSERT_TRUE(read_buf.ReadMany(&c, &i, &b, &s, &s2, &iv, &ia));
    EXPECT_EQ(c, kCharValue);
    EXPECT_EQ(i, kIntValue);
    EXPECT_EQ(b, kBooleanValue);
    EXPECT_EQ(s, kCharPtrValue);
    EXPECT_EQ(s2, kStringValue);
    EXPECT_EQ(iv, kIntVector);
    EXPECT_EQ(ia, kIntArray);
    ASSERT_TRUE(read_buf.Done());
  }
}

}  // namespace tachyon::base
