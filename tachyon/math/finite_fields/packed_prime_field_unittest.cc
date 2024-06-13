#include "gtest/gtest.h"

#include "tachyon/build/build_config.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

#if ARCH_CPU_X86_64
#include "tachyon/math/finite_fields/baby_bear/packed_baby_bear_avx2.h"
#include "tachyon/math/finite_fields/koala_bear/packed_koala_bear_avx2.h"
#include "tachyon/math/finite_fields/mersenne31/packed_mersenne31_avx2.h"
#if defined(TACHYON_HAS_AVX512)
#include "tachyon/math/finite_fields/baby_bear/packed_baby_bear_avx512.h"
#include "tachyon/math/finite_fields/koala_bear/packed_koala_bear_avx512.h"
#include "tachyon/math/finite_fields/mersenne31/packed_mersenne31_avx512.h"
#endif
#elif ARCH_CPU_ARM64
#include "tachyon/math/finite_fields/baby_bear/packed_baby_bear_neon.h"
#include "tachyon/math/finite_fields/koala_bear/packed_koala_bear_neon.h"
#include "tachyon/math/finite_fields/mersenne31/packed_mersenne31_neon.h"
#endif

namespace tachyon::math {

namespace {

template <typename PackedPrimeField>
class PackedPrimeFieldTest : public FiniteFieldTest<PackedPrimeField> {};

}  // namespace

using PackedPrimeFieldTypes = testing::Types<
#if ARCH_CPU_X86_64
    PackedBabyBearAVX2, PackedMersenne31AVX2, PackedKoalaBearAVX2
#if defined(TACHYON_HAS_AVX512)
    ,
    PackedBabyBearAVX512, PackedMersenne31AVX512, PackedKoalaBearAVX512
#endif
#elif ARCH_CPU_ARM64
    PackedBabyBearNeon, PackedMersenne31Neon, PackedKoalaBearNeon
#endif
    >;

TYPED_TEST_SUITE(PackedPrimeFieldTest, PackedPrimeFieldTypes);

TYPED_TEST(PackedPrimeFieldTest, Zero) {
  using PackedPrimeField = TypeParam;

  PackedPrimeField zero = PackedPrimeField::Zero();
  EXPECT_TRUE(zero.IsZero());
  for (size_t i = 0; i < PackedPrimeField::N; ++i) {
    EXPECT_TRUE(zero[i].IsZero());
  }
  EXPECT_FALSE(PackedPrimeField::Random().IsZero());
}

TYPED_TEST(PackedPrimeFieldTest, One) {
  using PackedPrimeField = TypeParam;

  PackedPrimeField one = PackedPrimeField::One();
  EXPECT_TRUE(one.IsOne());
  for (size_t i = 0; i < PackedPrimeField::N; ++i) {
    EXPECT_TRUE(one[i].IsOne());
  }
  EXPECT_FALSE(PackedPrimeField::Random().IsOne());
}

TYPED_TEST(PackedPrimeFieldTest, Broadcast) {
  using PackedPrimeField = TypeParam;
  using PrimeField = typename PackedPrimeField::PrimeField;

  PrimeField r = PrimeField::Random();
  PackedPrimeField f = PackedPrimeField::Broadcast(r);
  for (size_t i = 0; i < PackedPrimeField::N; ++i) {
    EXPECT_EQ(f[i], r);
  }
}

TYPED_TEST(PackedPrimeFieldTest, Random) {
  using PackedPrimeField = TypeParam;

  bool success = false;
  PackedPrimeField r = PackedPrimeField::Random();
  for (size_t i = 0; i < 100; ++i) {
    if (r != PackedPrimeField::Random()) {
      success = true;
      break;
    }
  }
  EXPECT_TRUE(success);
}

TYPED_TEST(PackedPrimeFieldTest, Add) {
  using PackedPrimeField = TypeParam;

  PackedPrimeField a = PackedPrimeField::Random();
  PackedPrimeField b = PackedPrimeField::Random();
  PackedPrimeField zero = PackedPrimeField::Zero();

  struct {
    PackedPrimeField a;
    PackedPrimeField b;
  } tests[] = {
      {a, b},
      {a, zero},
  };

  for (auto& test : tests) {
    PackedPrimeField c = test.a + test.b;
    for (size_t i = 0; i < PackedPrimeField::N; ++i) {
      ASSERT_EQ(c[i], test.a[i] + test.b[i]);
    }
    test.a += test.b;
    for (size_t i = 0; i < PackedPrimeField::N; ++i) {
      EXPECT_EQ(c[i], test.a[i]);
    }
  }
}

TYPED_TEST(PackedPrimeFieldTest, Sub) {
  using PackedPrimeField = TypeParam;

  PackedPrimeField a = PackedPrimeField::Random();
  PackedPrimeField b = PackedPrimeField::Random();
  PackedPrimeField zero = PackedPrimeField::Zero();

  struct {
    PackedPrimeField a;
    PackedPrimeField b;
  } tests[] = {
      {a, b},
      {a, zero},
  };

  for (auto& test : tests) {
    PackedPrimeField c = test.a - test.b;
    for (size_t i = 0; i < PackedPrimeField::N; ++i) {
      ASSERT_EQ(c[i], test.a[i] - test.b[i]);
    }
    test.a -= test.b;
    for (size_t i = 0; i < PackedPrimeField::N; ++i) {
      EXPECT_EQ(c[i], test.a[i]);
    }
  }
}

TYPED_TEST(PackedPrimeFieldTest, Negate) {
  using PackedPrimeField = TypeParam;

  PackedPrimeField a = PackedPrimeField::Random();
  PackedPrimeField zero = PackedPrimeField::Zero();

  struct {
    PackedPrimeField a;
  } tests[] = {
      {a},
      {zero},
  };

  for (auto& test : tests) {
    PackedPrimeField c = -test.a;
    for (size_t i = 0; i < PackedPrimeField::N; ++i) {
      ASSERT_EQ(c[i], -test.a[i]);
    }
    test.a.NegateInPlace();
    for (size_t i = 0; i < PackedPrimeField::N; ++i) {
      EXPECT_EQ(c[i], test.a[i]);
    }
  }
}

TYPED_TEST(PackedPrimeFieldTest, Mul) {
  using PackedPrimeField = TypeParam;

  PackedPrimeField a = PackedPrimeField::Random();
  PackedPrimeField b = PackedPrimeField::Random();
  PackedPrimeField zero = PackedPrimeField::Zero();
  PackedPrimeField one = PackedPrimeField::One();

  struct {
    PackedPrimeField a;
    PackedPrimeField b;
  } tests[] = {
      {a, b},
      {a, zero},
      {a, one},
  };

  for (auto& test : tests) {
    PackedPrimeField c = test.a * test.b;
    for (size_t i = 0; i < PackedPrimeField::N; ++i) {
      ASSERT_EQ(c[i], test.a[i] * test.b[i]);
    }
    test.a *= test.b;
    for (size_t i = 0; i < PackedPrimeField::N; ++i) {
      EXPECT_EQ(c[i], test.a[i]);
    }
  }
}

TYPED_TEST(PackedPrimeFieldTest, Inverse) {
  using PackedPrimeField = TypeParam;

  PackedPrimeField a = PackedPrimeField::Random();
  PackedPrimeField zero = PackedPrimeField::Zero();

  struct {
    PackedPrimeField a;
  } tests[] = {
      {a},
      {zero},
  };

  for (auto& test : tests) {
    std::optional<PackedPrimeField> c = test.a.Inverse();
    if (test.a.IsZero()) {
      for (size_t i = 0; i < PackedPrimeField::N; ++i) {
        EXPECT_TRUE((*c)[i].IsZero());
      }
    } else {
      for (size_t i = 0; i < PackedPrimeField::N; ++i) {
        EXPECT_EQ((*c)[i], test.a[i].Inverse());
      }
    }
    ASSERT_TRUE(test.a.InverseInPlace());
    for (size_t i = 0; i < PackedPrimeField::N; ++i) {
      EXPECT_EQ((*c)[i], test.a[i]);
    }
  }
}

}  // namespace tachyon::math
