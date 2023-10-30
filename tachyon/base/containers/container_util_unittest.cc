#include "tachyon/base/containers/container_util.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace tachyon::base {

constexpr size_t kCount = 1000;

TEST(ContainerUtilTest, CreateRangedVector) {
  struct {
    int start;
    int end;
    int step;
    std::vector<int> answers;
  } tests[] = {
      {0, 9, 3, {0, 3, 6}},
      {1, 9, 3, {1, 4, 7}},
  };

  for (const auto& test : tests) {
    auto ranges = CreateRangedVector(test.start, test.end, test.step);
    EXPECT_THAT(ranges, testing::ContainerEq(test.answers));
  }
}

TEST(ContainerUtilTest, CreateVectorWithGenerator) {
  EXPECT_THAT(CreateVector(5, ([]() { return 3; })),
              testing::ContainerEq(std::vector<int>{3, 3, 3, 3, 3}));
  EXPECT_THAT(CreateVector(5, ([](int idx) { return idx + 1; })),
              testing::ContainerEq(std::vector<int>{1, 2, 3, 4, 5}));
}

TEST(ContainerUtilTest, CreateVectorWithInitialValue) {
  EXPECT_THAT(CreateVector(5, 1),
              testing::ContainerEq(std::vector<int>{1, 1, 1, 1, 1}));
}

TEST(ContainerUtilTest, Map) {
  std::vector<int> arr({1, 2, 3});
  EXPECT_THAT(Map(arr.begin(), arr.end(),
                  [](int v) { return static_cast<double>(v * 2); }),
              testing::ContainerEq(std::vector<double>{2.0, 4.0, 6.0}));
  EXPECT_THAT(Map(arr, [](int v) { return static_cast<double>(v * 2); }),
              testing::ContainerEq(std::vector<double>{2.0, 4.0, 6.0}));
}

TEST(ContainerUtilTest, FlatMap) {
  std::vector<int> arr({1, 2});
  arr = base::FlatMap(arr, [](int value) {
    return std::vector<int>({
        value,
        value + 1,
        value + 2,
    });
  });
  EXPECT_THAT(arr, testing::ContainerEq(std::vector<int>{1, 2, 3, 2, 3, 4}));
}

TEST(ContainerUtilTest, FindIndex) {
  std::vector<int> arr({1, 2});
  std::optional<size_t> index = base::FindIndex(arr, 1);
  EXPECT_EQ(index.value(), 0);
  index = base::FindIndex(arr, 3);
  EXPECT_FALSE(index.has_value());
}

TEST(ContainerUtilTest, FindIndexIf) {
  std::vector<int> arr({1, 2});
  std::optional<size_t> index =
      base::FindIndexIf(arr, [](size_t v) { return v == 1; });
  EXPECT_EQ(index.value(), 0);
  index = base::FindIndexIf(arr, [](size_t v) { return v == 3; });
  EXPECT_FALSE(index.has_value());
}

TEST(ContainerUtilTest, Shuffle) {
  std::vector<int> vec = {1, 2, 3};
  std::vector<int> vec2 = vec;
  for (size_t i = 0; i < kCount; ++i) {
    Shuffle(vec2);
    if (vec != vec2) {
      SUCCEED();
      return;
    }
  }
  FAIL() << "shuffle seems not working";
}

}  // namespace tachyon::base
