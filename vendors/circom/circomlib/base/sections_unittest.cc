#include "circomlib/base/sections.h"

#include "gtest/gtest.h"

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/base/logging.h"

namespace tachyon::circom {

namespace {

enum class Type {
  kDummy,
  kDummy2,
};

std::string_view TypeToString(Type type) {
  switch (type) {
    case Type::kDummy:
      return "Dummy";
    case Type::kDummy2:
      return "Dummy2";
  }
  NOTREACHED();
  return "";
}

}  // namespace

TEST(SectionsTest, ReadAndGet) {
  {
    base::Uint8VectorBuffer buffer;
    Sections<Type> sections(buffer, &TypeToString);
    // Should return false when it fails to read the number of sections.
    ASSERT_FALSE(sections.Read());
  }

  {
    base::Uint8VectorBuffer buffer;
    ASSERT_TRUE(buffer.Write(uint32_t{1}));
    buffer.set_buffer_offset(0);
    Sections<Type> sections(buffer, &TypeToString);
    // Should return false when it fails to read the section type.
    ASSERT_FALSE(sections.Read());
  }

  {
    base::Uint8VectorBuffer buffer;
    ASSERT_TRUE(buffer.Write(uint32_t{1}));
    ASSERT_TRUE(buffer.Write(Type::kDummy));
    buffer.set_buffer_offset(0);
    Sections<Type> sections(buffer, &TypeToString);
    // Should return false when it fails to read the section size.
    ASSERT_FALSE(sections.Read());
  }

  {
    base::Uint8VectorBuffer buffer;
    ASSERT_TRUE(buffer.Write(uint32_t{1}));
    ASSERT_TRUE(buffer.Write(Type::kDummy));
    ASSERT_TRUE(buffer.Write(uint64_t{32}));
    size_t expected = buffer.buffer_offset();
    buffer.set_buffer_offset(0);
    Sections<Type> sections(buffer, &TypeToString);
    ASSERT_TRUE(sections.Read());

    ASSERT_FALSE(sections.MoveTo(Type::kDummy2));
    ASSERT_TRUE(sections.MoveTo(Type::kDummy));
    EXPECT_EQ(buffer.buffer_offset(), expected);
  }
}

}  // namespace tachyon::circom
