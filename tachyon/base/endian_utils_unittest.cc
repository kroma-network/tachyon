#include "tachyon/base/endian_utils.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/base/containers/cxx20_erase_vector.h"

struct Range {
  size_t start;
  size_t end;
};

TEST(EndianUtils, ForFromBiggest) {
  struct {
    size_t start;
    size_t end;
    std::vector<size_t> answers;
  } tests[] = {
      {0, 5, {4, 3, 2, 1, 0}},
      {2, 4, {3, 2}},
  };

  for (const auto& test : tests) {
    std::vector<size_t> idxs;
    FOR_FROM_BIGGEST(i, test.start, test.end) { idxs.push_back(i); }

    EXPECT_THAT(idxs, testing::ContainerEq(test.answers));
  }
}

TEST(EndianUtils, ForFromSmallest) {
  struct {
    size_t start;
    size_t end;
    std::vector<size_t> answers;
  } tests[] = {
      {0, 5, {0, 1, 2, 3, 4}},
      {2, 4, {2, 3}},
  };

  for (const auto& test : tests) {
    std::vector<size_t> idxs;
    FOR_FROM_SMALLEST(i, test.start, test.end) { idxs.push_back(i); }

    EXPECT_THAT(idxs, testing::ContainerEq(test.answers));
  }
}

TEST(EndianUtils, ForBugSmallest) {
  std::vector<size_t> idxs;
  FOR_BUT_SMALLEST(i, 5) { idxs.push_back(i); }

  std::vector<size_t> answers({0, 1, 2, 3, 4});
  tachyon::base::Erase(answers, SMALLEST_INDEX(5));
  EXPECT_THAT(idxs, testing::ContainerEq(answers));
}

TEST(EndianUtils, SmallestIndex) {
#if ARCH_CPU_BIG_ENDIAN
  int answer = 4;
#else  // ARCH_CPU_LITTLE_ENDIAN
  int answer = 0;
#endif
  EXPECT_EQ(SMALLEST_INDEX(5), answer);
}

TEST(EndianUtils, BiggestIndex) {
#if ARCH_CPU_BIG_ENDIAN
  int answer = 0;
#else  // ARCH_CPU_LITTLE_ENDIAN
  int answer = 4;
#endif
  EXPECT_EQ(BIGGEST_INDEX(5), answer);
}
