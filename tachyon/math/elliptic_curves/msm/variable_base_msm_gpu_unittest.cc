#include "tachyon/math/elliptic_curves/msm/variable_base_msm_gpu.h"

#include <limits>

#include "gtest/gtest.h"

#include "tachyon/device/gpu/gpu_enums.h"
#include "tachyon/device/gpu/scoped_mem_pool.h"
#include "tachyon/device/gpu/scoped_stream.h"
#include "tachyon/math/elliptic_curves/msm/test/variable_base_msm_test_set.h"

namespace tachyon::math {

namespace {

constexpr size_t kThreadNum = 32;

using namespace device;

template <typename Point>
class VariableMSMCorrectnessGpuTest : public testing::Test {
 public:
  using Curve = typename Point::Curve;

  constexpr static size_t kLogCount = 10;
  constexpr static size_t kCount = 1 << kLogCount;

  static void SetUpTestSuite() {
    Point::Curve::Init();

    test_set_ = VariableBaseMSMTestSet<Point>::Random(
        kCount, VariableBaseMSMMethod::kMSM);

    expected_ = test_set_.answer.ToProjective();
  }

  static void TearDownTestSuite() { GPU_MUST_SUCCESS(gpuDeviceReset(), ""); }

 protected:
  static VariableBaseMSMTestSet<Point> test_set_;
  static ProjectivePoint<Curve> expected_;
};

template <typename Point>
VariableBaseMSMTestSet<Point> VariableMSMCorrectnessGpuTest<Point>::test_set_;
template <typename Point>
ProjectivePoint<typename Point::Curve>
    VariableMSMCorrectnessGpuTest<Point>::expected_;

}  // namespace

using PointTypes = testing::Types<bn254::G1AffinePoint, bn254::G2AffinePoint>;
TYPED_TEST_SUITE(VariableMSMCorrectnessGpuTest, PointTypes);

TYPED_TEST(VariableMSMCorrectnessGpuTest, MSM) {
  using Point = TypeParam;
  using Curve = typename Point::Curve;

  gpuMemPoolProps props = {gpuMemAllocationTypePinned,
                           gpuMemHandleTypeNone,
                           {gpuMemLocationTypeDevice, 0}};
  gpu::ScopedMemPool mem_pool = gpu::CreateMemPool(&props);

  uint64_t mem_pool_threshold = std::numeric_limits<uint64_t>::max();
  gpuError_t error = gpuMemPoolSetAttribute(
      mem_pool.get(), gpuMemPoolAttrReleaseThreshold, &mem_pool_threshold);
  ASSERT_EQ(error, gpuSuccess);

  gpu::ScopedStream stream = gpu::CreateStream();

  VariableBaseMSMGpu<Point> msm_gpu(mem_pool.get(), stream.get());
  ProjectivePoint<Curve> actual;
  ASSERT_TRUE(
      msm_gpu.Run(this->test_set_.bases, this->test_set_.scalars, &actual));
  EXPECT_EQ(actual, this->expected_);
}

}  // namespace tachyon::math
