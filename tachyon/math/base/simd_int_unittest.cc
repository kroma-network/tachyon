#include "tachyon/math/base/simd_int.h"

#include "gtest/gtest.h"

namespace tachyon::math {

template <typename SimdInt>
class SimdIntTest : public testing::Test {};

using SimdIntTypes = testing::Types<SimdInt128
#if ARCH_CPU_X86_64
                                    ,
                                    SimdInt256
#endif
                                    >;
TYPED_TEST_SUITE(SimdIntTest, SimdIntTypes);

TYPED_TEST(SimdIntTest, Zero) {
  using SimdInt = TypeParam;

  EXPECT_TRUE(SimdInt().IsZero());
  EXPECT_TRUE(SimdInt::Zero().IsZero());
  EXPECT_FALSE(SimdInt::One().IsZero());
  EXPECT_FALSE(SimdInt::Max().IsZero());
}

TYPED_TEST(SimdIntTest, One) {
  using SimdInt = TypeParam;

  EXPECT_FALSE(SimdInt().IsOne());
  EXPECT_FALSE(SimdInt::Zero().IsOne());
  EXPECT_TRUE(SimdInt::One().IsOne());
  EXPECT_FALSE(SimdInt::Max().IsOne());
}

TYPED_TEST(SimdIntTest, Max) {
  using SimdInt = TypeParam;

  EXPECT_FALSE(SimdInt().IsMax());
  EXPECT_FALSE(SimdInt::Zero().IsMax());
  EXPECT_FALSE(SimdInt::One().IsMax());
  EXPECT_TRUE(SimdInt::Max().IsMax());
}

template <typename SimdInt, typename T>
void TestBroadcast() {
  using BigInt = typename SimdInt::value_type;

  T v = base::Uniform(base::Range<T>());
  BigInt expected;
  for (size_t i = 0; i < BigInt::kByteNums / sizeof(T); ++i) {
    reinterpret_cast<T*>(&expected)[i] = v;
  }
  BigInt actual = SimdInt::Broadcast(v).value();
  EXPECT_EQ(actual, expected);
}

TYPED_TEST(SimdIntTest, Broadcast) {
  using SimdInt = TypeParam;

  TestBroadcast<SimdInt, uint8_t>();
  TestBroadcast<SimdInt, uint16_t>();
  TestBroadcast<SimdInt, uint32_t>();
  TestBroadcast<SimdInt, uint64_t>();
}

TYPED_TEST(SimdIntTest, EqualityOperations) {
  using SimdInt = TypeParam;
  using BigInt = typename SimdInt::value_type;

  SimdInt a = SimdInt::Random();
  SimdInt b = SimdInt(a.value() + BigInt(1));
  EXPECT_EQ(a, a);
  EXPECT_NE(a, b);
}

TYPED_TEST(SimdIntTest, BitOperations) {
  using SimdInt = TypeParam;
  using BigInt = typename SimdInt::value_type;

  SimdInt a = SimdInt::Random();
  SimdInt not_a = !a;
  EXPECT_EQ(a & a, a);
  EXPECT_TRUE((a & not_a).IsZero());
  EXPECT_EQ(a | a, a);
  EXPECT_TRUE((a | not_a).IsMax());
  EXPECT_TRUE((a ^ a).IsZero());
  EXPECT_TRUE((a ^ not_a).IsMax());

  size_t count = base::Uniform(base::Range<size_t>(0, SimdInt::kBits));
  BigInt expected = a.value() << count;
  EXPECT_EQ((a << count).value(), expected);

  expected = a.value() >> count;
  EXPECT_EQ((a >> count).value(), expected);
}

}  // namespace tachyon::math
