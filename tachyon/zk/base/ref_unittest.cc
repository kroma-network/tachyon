#include "tachyon/zk/base/ref.h"

#include "gtest/gtest.h"

namespace tachyon::zk {

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

}  // namespace tachyon::zk
