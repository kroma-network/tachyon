#include "absl/strings/substitute.h"
#include "gtest/gtest.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/device/gpu/gpu_enums.h"
#include "tachyon/device/gpu/scoped_async_memory.h"
#include "tachyon/device/gpu/scoped_mem_pool.h"
#include "tachyon/device/gpu/scoped_memory.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1_cuda.cu.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm_cuda.cu.h"

namespace tachyon::math {

namespace {

constexpr size_t kThreadNum = 32;

using namespace device;

class VariableMSMCorrectnessCudaTest : public testing::Test {
 public:
  constexpr static size_t kLogCount = 10;
  constexpr static size_t kCount = 1 << kLogCount;

  static void SetUpTestSuite() {
    bn254::G1AffinePoint::Curve::Init();
    VariableBaseMSMCuda<bn254::G1AffinePointCuda::Curve>::Setup();

    std::vector<bn254::G1AffinePoint> bases = base::CreateVector(
        kCount, []() { return bn254::G1AffinePoint::Random(); });
    std::vector<bn254::Fr> scalars =
        base::CreateVector(kCount, []() { return bn254::Fr::Random(); });

    expected_ = VariableBaseMSM<bn254::G1JacobianPoint>::MSM(
        base::Map(bases,
                  [](const bn254::G1AffinePoint& p) { return p.ToJacobian(); }),
        scalars);

    d_bases_ = gpu::Malloc<bn254::G1AffinePointCuda>(kCount);
    d_scalars_ = gpu::Malloc<bn254::FrCuda>(kCount);
    d_results_ = gpu::Malloc<bn254::G1JacobianPointCuda>(256);
    u_results_.reset(new bn254::G1JacobianPoint[256]);

    gpuMemcpy(d_bases_.get(), bases.data(),
              sizeof(bn254::G1AffinePointCuda) * kCount, gpuMemcpyHostToDevice);
    gpuMemcpy(d_scalars_.get(), scalars.data(), sizeof(bn254::FrCuda) * kCount,
              gpuMemcpyHostToDevice);
  }

  static void TearDownTestSuite() {
    d_bases_.reset();
    d_scalars_.reset();
    d_results_.reset();

    gpuError_t error = gpuDeviceReset();
    GPU_CHECK(error == gpuSuccess, error);
  }

 protected:
  static gpu::ScopedDeviceMemory<bn254::G1AffinePointCuda> d_bases_;
  static gpu::ScopedDeviceMemory<bn254::FrCuda> d_scalars_;
  static gpu::ScopedDeviceMemory<bn254::G1JacobianPointCuda> d_results_;
  static std::unique_ptr<bn254::G1JacobianPoint[]> u_results_;
  static bn254::G1JacobianPoint expected_;
};

gpu::ScopedDeviceMemory<bn254::G1AffinePointCuda>
    VariableMSMCorrectnessCudaTest::d_bases_;
gpu::ScopedDeviceMemory<bn254::FrCuda>
    VariableMSMCorrectnessCudaTest::d_scalars_;
gpu::ScopedDeviceMemory<bn254::G1JacobianPointCuda>
    VariableMSMCorrectnessCudaTest::d_results_;
std::unique_ptr<bn254::G1JacobianPoint[]>
    VariableMSMCorrectnessCudaTest::u_results_;
bn254::G1JacobianPoint VariableMSMCorrectnessCudaTest::expected_;

}  // namespace

TEST_F(VariableMSMCorrectnessCudaTest, MSM) {
  gpuMemPoolProps props = {gpuMemAllocationTypePinned,
                           gpuMemHandleTypeNone,
                           {gpuMemLocationTypeDevice, 0}};
  gpu::ScopedMemPool mem_pool = gpu::CreateMemPool(&props);
  uint64_t mem_pool_threshold = std::numeric_limits<uint64_t>::max();
  gpuError_t error = gpuMemPoolSetAttribute(
      mem_pool.get(), gpuMemPoolAttrReleaseThreshold, &mem_pool_threshold);
  ASSERT_EQ(error, gpuSuccess);

  gpu::ScopedStream stream = gpu::CreateStream();
  kernels::msm::ExecutionConfig<bn254::G1AffinePointCuda::Curve> config;
  config.mem_pool = mem_pool.get();
  config.stream = stream.get();
  config.bases = d_bases_.get();
  config.scalars = d_scalars_.get();
  config.results = d_results_.get();
  config.log_scalars_count = kLogCount;

  error = VariableBaseMSMCuda<bn254::G1AffinePointCuda::Curve>::ExecuteAsync(
      config);
  ASSERT_EQ(error, gpuSuccess);
  error = gpuStreamSynchronize(stream.get());
  ASSERT_EQ(error, gpuSuccess);

  gpuMemcpy(u_results_.get(), config.results,
            sizeof(bn254::G1JacobianPointCuda) * 256, gpuMemcpyDefault);
  bn254::G1JacobianPoint actual = bn254::G1JacobianPoint::Zero();
  for (size_t i = 0; i < bn254::Fr::Config::kModulusBits; ++i) {
    size_t index = bn254::Fr::Config::kModulusBits - i - 1;
    bn254::G1JacobianPoint bucket = u_results_[index];
    if (i == 0) {
      actual = bucket;
    } else {
      actual.DoubleInPlace();
      actual += bucket;
    }
  }
  EXPECT_EQ(actual, expected_);
}

}  // namespace tachyon::math
