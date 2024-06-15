#include "tachyon/base/containers/container_util.h"

#include <memory>
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

TEST(ContainerUtilTest, Map) {
  std::vector<int> arr({1, 2, 3});
  EXPECT_THAT(Map(arr.begin(), arr.end(),
                  [](int v) { return static_cast<double>(v * 2); }),
              testing::ContainerEq(std::vector<double>{2.0, 4.0, 6.0}));
  EXPECT_THAT(Map(arr, [](int v) { return static_cast<double>(v * 2); }),
              testing::ContainerEq(std::vector<double>{2.0, 4.0, 6.0}));
}

TEST(ContainerUtilTest, MapWithIdx) {
  std::vector<int> arr({1, 2, 3});
  EXPECT_THAT(
      Map(arr.begin(), arr.end(),
          [](size_t idx, int v) { return static_cast<double>(v * 2 + idx); }),
      testing::ContainerEq(std::vector<double>{2.0, 5.0, 8.0}));
  EXPECT_THAT(Map(arr, [](size_t idx,
                          int v) { return static_cast<double>(v * 2 + idx); }),
              testing::ContainerEq(std::vector<double>{2.0, 5.0, 8.0}));
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

TEST(ContainerUtilTest, DoFindIndices) {
  std::vector<int> arr({1, 2, 3, 1, 2, 3});
  std::vector<std::unique_ptr<std::input_iterator_tag>> tags;
  tags.push_back(std::make_unique<std::forward_iterator_tag>());
  tags.push_back(std::make_unique<std::random_access_iterator_tag>());
  for (const std::unique_ptr<std::input_iterator_tag>& tag : tags) {
    std::vector<size_t> indices =
        base::internal::DoFindIndices(arr.begin(), arr.end(), 1, *tag);
    EXPECT_EQ(indices, std::vector<size_t>({0, 3}));
    indices = base::internal::DoFindIndices(arr.begin(), arr.end(), 4, *tag);
    EXPECT_TRUE(indices.empty());
  }
}

TEST(ContainerUtilTest, FindIndices) {
  std::vector<int> arr({1, 2, 3, 1, 2, 3});
  std::vector<size_t> indices = base::FindIndices(arr, 1);
  EXPECT_EQ(indices, std::vector<size_t>({0, 3}));
  indices = base::FindIndices(arr, 4);
  EXPECT_TRUE(indices.empty());
}

TEST(ContainerUtilTest, DoFindIndicesIf) {
  std::vector<int> arr({1, 2, 3, 1, 2, 3});
  std::vector<std::unique_ptr<std::input_iterator_tag>> tags;
  tags.push_back(std::make_unique<std::forward_iterator_tag>());
  tags.push_back(std::make_unique<std::random_access_iterator_tag>());
  for (const std::unique_ptr<std::input_iterator_tag>& tag : tags) {
    std::vector<size_t> indices = base::internal::DoFindIndicesIf(
        arr.begin(), arr.end(), [](size_t v) { return v == 1; }, *tag);
    EXPECT_EQ(indices, std::vector<size_t>({0, 3}));
    indices = base::internal::DoFindIndicesIf(
        arr.begin(), arr.end(), [](size_t v) { return v == 4; }, *tag);
    EXPECT_TRUE(indices.empty());
  }
}

TEST(ContainerUtilTest, FindIndicesIf) {
  std::vector<int> arr({1, 2, 3, 1, 2, 3});
  std::vector<size_t> indices =
      base::FindIndicesIf(arr, [](size_t v) { return v == 1; });
  EXPECT_EQ(indices, std::vector<size_t>({0, 3}));
  indices = base::FindIndicesIf(arr, [](size_t v) { return v == 4; });
  EXPECT_TRUE(indices.empty());
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

TEST(ContainerUtilTest, BinarySearchByKey) {
  struct TestData {
    std::vector<int> data;
    int value;
    std::optional<size_t> expected_index;
  };

  std::vector<TestData> tests = {
      {{1, 2, 3, 4, 5}, 3, 2}, {{1, 2, 3, 4, 5}, 1, 0},
      {{1, 2, 3, 4, 5}, 5, 4}, {{1, 2, 3, 4, 5}, 6, std::nullopt},
      {{}, 3, std::nullopt},
  };

  for (const auto& test : tests) {
    auto it =
        BinarySearchByKey(test.data.begin(), test.data.end(), test.value,
                          [](int elem, int value) { return elem < value; });

    if (test.expected_index.has_value()) {
      EXPECT_EQ(std::distance(test.data.begin(), it),
                test.expected_index.value());
    } else {
      EXPECT_EQ(it, test.data.end());
    }
  }
}

}  // namespace tachyon::base
