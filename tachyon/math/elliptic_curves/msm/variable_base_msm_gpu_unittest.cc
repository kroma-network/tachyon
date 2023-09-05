#include "tachyon/math/elliptic_curves/msm/variable_base_msm_gpu.h"

#include "absl/strings/substitute.h"
#include "gtest/gtest.h"

#include "tachyon/device/gpu/gpu_enums.h"
#include "tachyon/device/gpu/gpu_memory.h"
#include "tachyon/device/gpu/scoped_mem_pool.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1_gpu.h"
#include "tachyon/math/elliptic_curves/msm/test/msm_test_set.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"

namespace tachyon::math {

namespace {

constexpr size_t kThreadNum = 32;

using namespace device;

class VariableMSMCorrectnessGpuTest : public testing::Test {
 public:
  constexpr static size_t kLogCount = 10;
  constexpr static size_t kCount = 1 << kLogCount;

  static void SetUpTestSuite() {
    bn254::G1AffinePoint::Curve::Init();
    VariableBaseMSMGpu<bn254::G1AffinePointGpu::Curve>::Setup();

    MSMTestSet<bn254::G1AffinePoint> test_set =
        MSMTestSet<bn254::G1AffinePoint>::Random(kCount, MSMMethod::kMSM);

    d_bases_ = gpu::GpuMemory<bn254::G1AffinePointGpu>::Malloc(kCount);
    d_scalars_ = gpu::GpuMemory<bn254::FrGpu>::Malloc(kCount);
    size_t bit_size = bn254::FrGpu::kModulusBits;
    d_results_ = gpu::GpuMemory<bn254::G1JacobianPointGpu>::Malloc(bit_size);
    u_results_.reset(new bn254::G1JacobianPoint[bit_size]);

    CHECK(d_bases_.CopyFrom(test_set.bases.data(), gpu::GpuMemoryType::kHost));
    CHECK(d_scalars_.CopyFrom(test_set.scalars.data(),
                              gpu::GpuMemoryType::kHost));
    expected_ = std::move(test_set.answer);
  }

  static void TearDownTestSuite() {
    d_bases_.reset();
    d_scalars_.reset();
    d_results_.reset();

    GPU_MUST_SUCCESS(gpuDeviceReset(), "");
  }

 protected:
  static gpu::GpuMemory<bn254::G1AffinePointGpu> d_bases_;
  static gpu::GpuMemory<bn254::FrGpu> d_scalars_;
  static gpu::GpuMemory<bn254::G1JacobianPointGpu> d_results_;
  static std::unique_ptr<bn254::G1JacobianPoint[]> u_results_;
  static bn254::G1JacobianPoint expected_;
};

gpu::GpuMemory<bn254::G1AffinePointGpu> VariableMSMCorrectnessGpuTest::d_bases_;
gpu::GpuMemory<bn254::FrGpu> VariableMSMCorrectnessGpuTest::d_scalars_;
gpu::GpuMemory<bn254::G1JacobianPointGpu>
    VariableMSMCorrectnessGpuTest::d_results_;
std::unique_ptr<bn254::G1JacobianPoint[]>
    VariableMSMCorrectnessGpuTest::u_results_;
bn254::G1JacobianPoint VariableMSMCorrectnessGpuTest::expected_;

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
  msm::ExecutionConfig<bn254::G1AffinePointGpu::Curve> config;
  config.mem_pool = mem_pool.get();
  config.stream = stream.get();
  config.bases = d_bases_.get();
  config.scalars = d_scalars_.get();
  config.results = d_results_.get();
  config.log_scalars_count = kLogCount;

  bn254::G1JacobianPoint actual;
  error = VariableBaseMSMGpu<bn254::G1AffinePointGpu::Curve>::Execute(
      config, u_results_.get(), &actual);
  ASSERT_EQ(error, gpuSuccess);
  EXPECT_EQ(actual, expected_);
}

}  // namespace tachyon::math
