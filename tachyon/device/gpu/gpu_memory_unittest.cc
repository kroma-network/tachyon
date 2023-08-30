#include "tachyon/device/gpu/gpu_memory.h"

#include "gtest/gtest.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/random.h"
#include "tachyon/device/gpu/scoped_mem_pool.h"

namespace tachyon::device::gpu {

TEST(GpuMemoryTest, Malloc) {
  auto memory = GpuMemory<int>::Malloc(512);
  gpuPointerAttributes attr;
  ASSERT_TRUE(memory.GetAttributes(&attr));
  ASSERT_EQ(attr.type, gpuMemoryTypeDevice);
}

TEST(GpuMemoryTest, MallocHost) {
  auto memory = GpuMemory<int>::MallocHost(512);
  gpuPointerAttributes attr;
  ASSERT_TRUE(memory.GetAttributes(&attr));
  ASSERT_EQ(attr.type, gpuMemoryTypeHost);
}

#if TACHYON_CUDA
TEST(GpuMemoryTest, MallocManaged) {
  auto memory = GpuMemory<int>::MallocManaged(512);
  gpuPointerAttributes attr;
  ASSERT_TRUE(memory.GetAttributes(&attr));
  ASSERT_EQ(attr.type, gpuMemoryTypeManaged);
}
#endif  // TACHYON_CUDA

TEST(GpuMemoryTest, Memset) {
  auto memory = GpuMemory<uint8_t>::Malloc(512);
  int value = 3;
  size_t from = 10;
  size_t len = 400;

  ASSERT_TRUE(memory.Memset(value, from, len));

  std::vector<uint8_t> vec;
  vec.resize(512);
  memory.CopyTo(vec.data(), GpuMemoryType::kHost);

  GPU_MUST_SUCCESS(gpuDeviceSynchronize(), "");

  for (size_t i = 0; i < vec.size(); ++i) {
    if (i >= from && i < (from + len)) {
      ASSERT_EQ(vec[i], value);
    } else {
      ASSERT_EQ(vec[i], 0);
    }
  }
}

TEST(GpuMemoryTest, CopyFrom) {
  std::vector<GpuMemory<int>> memories;
  memories.push_back(GpuMemory<int>::Malloc(512));
  memories.push_back(GpuMemory<int>::MallocHost(512));
#if TACHYON_CUDA
  memories.push_back(GpuMemory<int>::MallocManaged(512));
#endif  // TACHYON_CUDA

  {
    std::vector<int> host_memory = base::CreateVector(512, []() {
      return base::Uniform(static_cast<int>(0),
                           std::numeric_limits<int>::max());
    });
    std::vector<std::vector<int>> results;
    results.resize(memories.size());
    for (size_t i = 0; i < memories.size(); ++i) {
      ASSERT_TRUE(
          memories[i].CopyFrom(host_memory.data(), GpuMemoryType::kHost));
      ASSERT_TRUE(memories[i].ToStdVector(&results[i]));
    }

    GPU_MUST_SUCCESS(gpuDeviceSynchronize(), "");
    for (size_t i = 0; i < memories.size(); ++i) {
      EXPECT_EQ(results[i], host_memory);
    }
  }
  // TODO(chokobole): Create random device random memory
#if TACHYON_CUDA
  {
    auto unified_memory = GpuMemory<int>::MallocManaged(512);
    absl::Span<int> unified_memory_view(unified_memory.get(), 512);
    for (int& v : unified_memory_view) {
      v = base::Uniform(static_cast<int>(0), std::numeric_limits<int>::max());
    }
    std::vector<std::vector<int>> results;
    results.resize(memories.size());
    for (size_t i = 0; i < memories.size(); ++i) {
      ASSERT_TRUE(memories[i].CopyFrom(unified_memory_view.data(),
                                       GpuMemoryType::kUnified));
      ASSERT_TRUE(memories[i].ToStdVector(&results[i]));
    }

    GPU_MUST_SUCCESS(gpuDeviceSynchronize(), "");
    for (size_t i = 0; i < memories.size(); ++i) {
      EXPECT_EQ(absl::MakeConstSpan(results[i]), unified_memory_view);
    }
  }
#endif  // TACHYON_CUDA
}

}  // namespace tachyon::device::gpu
