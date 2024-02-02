#include "tachyon/zk/plonk/vanishing/value_source.h"

#include "gtest/gtest.h"

#include "tachyon/base/random.h"

namespace tachyon::zk::plonk {

#define TEST_CONSTANTS(name, index)                                       \
  size_t value = base::Uniform(base::Range<size_t>::From(index + 1));     \
                                                                          \
  EXPECT_TRUE(ValueSource::Constant(index).Is##name##Constant());         \
  EXPECT_FALSE(ValueSource::Constant(value).Is##name##Constant());        \
  EXPECT_FALSE(ValueSource::Intermediate(index).Is##name##Constant());    \
  EXPECT_FALSE(ValueSource::Fixed(index, index).Is##name##Constant());    \
  EXPECT_FALSE(ValueSource::Advice(index, index).Is##name##Constant());   \
  EXPECT_FALSE(ValueSource::Instance(index, index).Is##name##Constant()); \
  EXPECT_FALSE(ValueSource::Challenge(index).Is##name##Constant());       \
  EXPECT_FALSE(ValueSource::Beta().Is##name##Constant());                 \
  EXPECT_FALSE(ValueSource::Gamma().Is##name##Constant());                \
  EXPECT_FALSE(ValueSource::Theta().Is##name##Constant());                \
  EXPECT_FALSE(ValueSource::Y().Is##name##Constant());                    \
  EXPECT_FALSE(ValueSource::PreviousValue().Is##name##Constant())

TEST(ValueSourceTest, IsZeroConstant) {
  TEST_CONSTANTS(Zero, 0);
  EXPECT_TRUE(ValueSource::ZeroConstant().IsZeroConstant());
  EXPECT_FALSE(ValueSource::OneConstant().IsZeroConstant());
  EXPECT_FALSE(ValueSource::TwoConstant().IsZeroConstant());
}

TEST(ValueSourceTest, IsOneConstant) {
  TEST_CONSTANTS(One, 1);
  EXPECT_FALSE(ValueSource::ZeroConstant().IsOneConstant());
  EXPECT_TRUE(ValueSource::OneConstant().IsOneConstant());
  EXPECT_FALSE(ValueSource::TwoConstant().IsOneConstant());
}

TEST(ValueSourceTest, IsTwoConstant) {
  TEST_CONSTANTS(Two, 2);
  EXPECT_FALSE(ValueSource::ZeroConstant().IsTwoConstant());
  EXPECT_FALSE(ValueSource::OneConstant().IsTwoConstant());
  EXPECT_TRUE(ValueSource::TwoConstant().IsTwoConstant());
}

#undef TEST_CONSTANTS

}  // namespace tachyon::zk::plonk
