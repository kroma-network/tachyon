#include "tachyon/math/base/bit_iterator.h"

#include <vector>

#include "absl/strings/substitute.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace tachyon {
namespace math {

TEST(GmpUtilTest, BitIteratorLE) {
  struct {
    uint64_t value[1];
    std::vector<bool> answers;
  } tests[] = {
      {{0}, std::vector<bool>{}},
      {{2}, std::vector<bool>{0, 1}},
      {{4}, std::vector<bool>{0, 0, 1}},
      {{5}, std::vector<bool>{1, 0, 1}},
  };

  for (int i = 0; i < 2; ++i) {
    bool skip_trailing_zeros = i == 0;
    SCOPED_TRACE(
        absl::Substitute("skip_trailing_zeros: $0", skip_trailing_zeros));
    for (const auto& test : tests) {
      std::vector<bool> bits;
      auto it = BitIteratorLE<uint64_t[1]>::begin(&test.value);
      auto end =
          BitIteratorLE<uint64_t[1]>::end(&test.value, skip_trailing_zeros);
      while (it != end) {
        bits.push_back(*it);
        ++it;
      }

      if (skip_trailing_zeros) {
        EXPECT_THAT(bits, ::testing::ContainerEq(test.answers));
      } else {
        std::vector<bool> answers = test.answers;
        answers.reserve(64);
        std::fill_n(std::back_inserter(answers), 64 - test.answers.size(),
                    false);
        EXPECT_THAT(bits, ::testing::ContainerEq(answers));
      }
    }
  }
}

TEST(GmpUtilTest, BitIteratorBE) {
  struct {
    uint64_t value[1];
    std::vector<bool> answers;
  } tests[] = {
      {{0}, std::vector<bool>{}},
      {{2}, std::vector<bool>{1, 0}},
      {{4}, std::vector<bool>{1, 0, 0}},
      {{5}, std::vector<bool>{1, 0, 1}},
  };

  for (int i = 0; i < 2; ++i) {
    bool skip_leading_zeros = i == 0;
    SCOPED_TRACE(
        absl::Substitute("skip_leading_zeros: $0", skip_leading_zeros));
    for (const auto& test : tests) {
      std::vector<bool> bits;
      auto it =
          BitIteratorBE<uint64_t[1]>::begin(&test.value, skip_leading_zeros);
      auto end = BitIteratorBE<uint64_t[1]>::end(&test.value);
      while (it != end) {
        bits.push_back(*it);
        ++it;
      }

      if (skip_leading_zeros) {
        EXPECT_THAT(bits, ::testing::ContainerEq(test.answers));
      } else {
        std::vector<bool> answers;
        answers.reserve(64);
        std::fill_n(std::back_inserter(answers), 64 - test.answers.size(),
                    false);
        answers.insert(answers.end(), test.answers.begin(), test.answers.end());
        EXPECT_THAT(bits, ::testing::ContainerEq(answers));
      }
    }
  }
}

}  // namespace math
}  // namespace tachyon
