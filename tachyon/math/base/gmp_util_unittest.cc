#include "tachyon/math/base/gmp_util.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/base/logging.h"

namespace tachyon {
namespace math {

TEST(GmpUtilTest, BitIteratorLE) {
  struct {
    mpz_class value;
    std::vector<bool> answers;
  } tests[] = {
      {2, std::vector<bool>{0, 1}},
      {4, std::vector<bool>{0, 0, 1}},
      {5, std::vector<bool>{1, 0, 1}},
  };

  for (const auto& test : tests) {
    std::vector<bool> bits;
    auto it = gmp::BitIteratorLE::begin(&test.value);
    auto end = gmp::BitIteratorLE::end(&test.value);
    while (it != end) {
      bits.push_back(*it);
      ++it;
    }

    EXPECT_THAT(bits, ::testing::ContainerEq(test.answers));
  }
}

TEST(GmpUtilTest, BitIteratorLEWithZero) {
  std::vector<bool> bits;
  mpz_class value;
  auto it = gmp::BitIteratorLE::begin(&value);
  auto end = gmp::BitIteratorLE::end(&value);
  while (it != end) {
    bits.push_back(*it);
    ++it;
  }
  EXPECT_TRUE(bits.empty());
}

TEST(GmpUtilTest, BitIteratorBE) {
  struct {
    mpz_class value;
    std::vector<bool> answers;
  } tests[] = {
      {2, std::vector<bool>{1, 0}},
      {4, std::vector<bool>{1, 0, 0}},
      {5, std::vector<bool>{1, 0, 1}},
  };

  for (const auto& test : tests) {
    std::vector<bool> bits;
    auto it = gmp::BitIteratorBE::begin(&test.value);
    auto end = gmp::BitIteratorBE::end(&test.value);
    while (it != end) {
      bits.push_back(*it);
      ++it;
    }

    EXPECT_THAT(bits, ::testing::ContainerEq(test.answers));
  }
}

TEST(GmpUtilTest, BitIteratorBEWithZero) {
  std::vector<bool> bits;
  mpz_class value;
  auto it = gmp::BitIteratorBE::begin(&value);
  auto end = gmp::BitIteratorBE::end(&value);
  while (it != end) {
    bits.push_back(*it);
    ++it;
  }
  EXPECT_TRUE(bits.empty());
}

}  // namespace math
}  // namespace tachyon
