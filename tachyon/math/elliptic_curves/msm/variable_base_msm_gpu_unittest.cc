#include "tachyon/math/elliptic_curves/msm/variable_base_msm_gpu.h"

#include <limits>

#include "gtest/gtest.h"

#include "tachyon/device/gpu/gpu_enums.h"
#include "tachyon/device/gpu/gpu_memory.h"
#include "tachyon/device/gpu/scoped_mem_pool.h"
#include "tachyon/device/gpu/scoped_stream.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1_gpu.h"
#include "tachyon/math/elliptic_curves/msm/test/variable_base_msm_test_set.h"

namespace tachyon::math {

namespace {

constexpr size_t kThreadNum = 32;

using namespace device;

class VariableMSMCorrectnessGpuTest : public testing::Test {
 public:
  constexpr static size_t kLogCount = 10;
  constexpr static size_t kCount = 1 << kLogCount;

  static void SetUpTestSuite() {
    bn254::G1Curve::Init();

    VariableBaseMSMTestSet<bn254::G1AffinePoint> test_set =
        VariableBaseMSMTestSet<bn254::G1AffinePoint>::Random(
            kCount, VariableBaseMSMMethod::kMSM);

    d_bases_ = gpu::GpuMemory<bn254::G1AffinePointGpu>::Malloc(kCount);
    d_scalars_ = gpu::GpuMemory<bn254::FrGpu>::Malloc(kCount);

    CHECK(d_bases_.CopyFrom(test_set.bases.data(), gpu::GpuMemoryType::kHost));
    CHECK(d_scalars_.CopyFrom(test_set.scalars.data(),
                              gpu::GpuMemoryType::kHost));
    expected_ = test_set.answer.ToProjective();
  }

  static void TearDownTestSuite() {
    d_bases_.reset();
    d_scalars_.reset();

    GPU_MUST_SUCCESS(gpuDeviceReset(), "");
  }

 protected:
  static gpu::GpuMemory<bn254::G1AffinePointGpu> d_bases_;
  static gpu::GpuMemory<bn254::FrGpu> d_scalars_;
  static bn254::G1ProjectivePoint expected_;
};

gpu::GpuMemory<bn254::G1AffinePointGpu> VariableMSMCorrectnessGpuTest::d_bases_;
gpu::GpuMemory<bn254::FrGpu> VariableMSMCorrectnessGpuTest::d_scalars_;
bn254::G1ProjectivePoint VariableMSMCorrectnessGpuTest::expected_;

}  // namespace

TEST_F(VariableMSMCorrectnessGpuTest, MSM) {
  gpuMemPoolProps props = {gpuMemAllocationTypePinned,
                           gpuMemHandleTypeNone,
                           {gpuMemLocationTypeDevice, 0}};
  gpu::ScopedMemPool mem_pool = gpu::CreateMemPool(&props);

  uint64_t mem_pool_threshold = std::numeric_limits<uint64_t>::max();
  gpuError_t error = gpuMemPoolSetAttribute(
      mem_pool.get(), gpuMemPoolAttrReleaseThreshold, &mem_pool_threshold);
  ASSERT_EQ(error, gpuSuccess);

  gpu::ScopedStream stream = gpu::CreateStream();

  VariableBaseMSMGpu<bn254::G1CurveGpu> msm_gpu(mem_pool.get(), stream.get());
  bn254::G1ProjectivePoint actual;
  ASSERT_TRUE(msm_gpu.Run(d_bases_, d_scalars_, kCount, &actual));
  EXPECT_EQ(actual, expected_);
}

}  // namespace tachyon::math
