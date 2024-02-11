#include "tachyon/base/parallelize.h"

#include <functional>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/base/containers/container_util.h"

namespace tachyon::base {

namespace {

void IncrementEachElement(absl::Span<int> chunk) {
  for (int& v : chunk) {
    v += 1;
  }
}

int SumElements(absl::Span<int> chunk) {
  int ret = 0;
  for (int v : chunk) {
    ret += v;
  }
  return ret;
}

}  // namespace

TEST(ParallelizeTest, ParallelizeByChunks) {
  std::vector<int> test_in = {0, 1, 2, 3, 4, 5};
  std::vector<int> expected = {1, 2, 3, 4, 5, 6};
  size_t chunk_size = 2;

  ParallelizeByChunkSize(test_in, chunk_size,
                         [chunk_size](absl::Span<int> chunk) {
                           EXPECT_EQ(chunk.size(), chunk_size);
                           IncrementEachElement(chunk);
                         });
  EXPECT_EQ(expected, test_in);

  expected = {2, 3, 4, 5, 6, 7};
  std::vector<size_t> chunk_indices =
      base::CreateVector(test_in.size() / chunk_size, size_t{0});
  ParallelizeByChunkSize(
      test_in, chunk_size,
      [chunk_size, &chunk_indices](absl::Span<int> chunk, size_t chunk_offset) {
        EXPECT_EQ(chunk.size(), chunk_size);
        chunk_indices[chunk_offset] = chunk_offset;
        IncrementEachElement(chunk);
      });
  EXPECT_EQ(expected, test_in);
  EXPECT_THAT(chunk_indices, testing::UnorderedElementsAreArray({0, 1, 2}));

  expected = {3, 4, 5, 6, 7, 8};
  chunk_indices = base::CreateVector(test_in.size() / chunk_size, size_t{0});
  std::vector<size_t> chunk_sizes =
      base::CreateVector(test_in.size() / chunk_size, size_t{0});
  ParallelizeByChunkSize(
      test_in, chunk_size,
      [chunk_size, &chunk_indices, &chunk_sizes](
          absl::Span<int> chunk, size_t chunk_index, size_t chunk_size_in) {
        EXPECT_EQ(chunk.size(), chunk_size);
        chunk_indices[chunk_index] = chunk_index;
        chunk_sizes[chunk_index] = chunk_size_in;
        IncrementEachElement(chunk);
      });
  EXPECT_EQ(expected, test_in);
  EXPECT_THAT(chunk_indices, testing::UnorderedElementsAreArray({0, 1, 2}));
  EXPECT_THAT(chunk_sizes, testing::UnorderedElementsAreArray({2, 2, 2}));
}

TEST(ParallelizeTest, Parallelize) {
  std::vector<int> test_in = {0, 1, 2, 3, 4, 5};
  std::vector<int> expected = {1, 2, 3, 4, 5, 6};

  Parallelize(test_in, &IncrementEachElement);
  EXPECT_EQ(expected, test_in);

  expected = {2, 3, 4, 5, 6, 7};
  Parallelize(test_in, [](absl::Span<int> chunk, size_t chunk_idx) {
    IncrementEachElement(chunk);
  });
  EXPECT_EQ(expected, test_in);

  expected = {3, 4, 5, 6, 7, 8};
  Parallelize(test_in, [](absl::Span<int> chunk, size_t chunk_idx,
                          size_t chunk_size) { IncrementEachElement(chunk); });
  EXPECT_EQ(expected, test_in);
}

TEST(ParallelizeTest, ParallelizeMapByChunkSize) {
  std::vector<int> test_in = {0, 1, 2, 3, 4, 5};

  std::vector<int> values = ParallelizeMapByChunkSize(test_in, 2, &SumElements);
  EXPECT_THAT(values, testing::ElementsAreArray({1, 5, 9}));
}

TEST(ParallelizeTest, ParallelizeMap) {
  std::vector<int> test_in = {0, 1, 2, 3, 4, 5};

  std::vector<int> values = ParallelizeMap(test_in, &SumElements);
  EXPECT_EQ(std::accumulate(values.begin(), values.end(), 0, std::plus<int>()),
            15);
}

}  // namespace tachyon::base
