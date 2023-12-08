#include "tachyon/base/ref.h"

#include "absl/hash/hash_testing.h"
#include "gtest/gtest.h"

namespace tachyon::base {

TEST(RefTest, BoolCasting) {
  Ref<int> ref;
  EXPECT_FALSE(ref);
  int value = 1;
  ref = Ref<int>(&value);
  EXPECT_TRUE(ref);
}

TEST(RefTest, MutableRef) {
  int value = 1;
  Ref<int> ref(&value);
  EXPECT_EQ(*ref, 1);
  EXPECT_EQ(ref.get(), &value);
  *ref = 2;
  EXPECT_EQ(*ref, 2);
  EXPECT_EQ(ref.get(), &value);
}

TEST(RefTest, ConstRef) {
  int value = 1;
  Ref<const int> ref(&value);
  EXPECT_EQ(*ref, 1);
  EXPECT_EQ(ref.get(), &value);
}

TEST(RefTest, Shallow) {
  int value = 1;
  ShallowRef<int> ref(&value);
  int value2 = 1;
  ShallowRef<int> ref2(&value2);
  int value3 = 2;
  ShallowRef<int> ref3(&value3);
  EXPECT_TRUE(ref != ref2);
  EXPECT_FALSE(ref == ref2);
  EXPECT_TRUE(ref != ref3);
  EXPECT_FALSE(ref == ref3);

  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(
      std::make_tuple(ref, ref2, ref3)));
}

TEST(RefTest, Deep) {
  int value = 1;
  DeepRef<int> ref(&value);
  int value2 = 1;
  DeepRef<int> ref2(&value2);
  int value3 = 2;
  DeepRef<int> ref3(&value3);
  EXPECT_TRUE(ref == ref2);
  EXPECT_FALSE(ref != ref2);
  EXPECT_FALSE(ref == ref3);
  EXPECT_TRUE(ref != ref3);

  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(
      std::make_tuple(ref, ref2, ref3)));
}

}  // namespace tachyon::base
