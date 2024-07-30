#include "absl/strings/substitute.h"
#include "gtest/gtest.h"

#include "tachyon/device/gpu/gpu_memory.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1_gpu.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/kernels/elliptic_curve_ops.cu.h"
#include "tachyon/math/geometry/point_conversions.h"
#include "tachyon/math/test/launch_op_macros.h"

namespace tachyon::math {

namespace {

constexpr size_t kThreadNum = 32;

#define DEFINE_LAUNCH_FIELD_BINARY_OP(method)                          \
  DEFINE_LAUNCH_BINARY_OP(kThreadNum, method, bn254::G1AffinePointGpu, \
                          bn254::G1JacobianPointGpu)

DEFINE_LAUNCH_FIELD_BINARY_OP(Add)

DEFINE_LAUNCH_UNARY_OP(kThreadNum, Double, bn254::G1AffinePointGpu,
                       bn254::G1JacobianPointGpu)
DEFINE_LAUNCH_UNARY_OP(kThreadNum, Negate, bn254::G1AffinePointGpu,
                       bn254::G1AffinePointGpu)

#undef DEFINE_LAUNCH_FIELD_BINARY_OP

#define DEFINE_LAUNCH_COMPARISON_OP(method) \
  DEFINE_LAUNCH_BINARY_OP(kThreadNum, method, bn254::G1AffinePointGpu, bool)

DEFINE_LAUNCH_COMPARISON_OP(Eq)
DEFINE_LAUNCH_COMPARISON_OP(Ne)

#undef DEFINE_LAUNCH_COMPARISON_OP

using namespace device;

class AffinePointCorrectnessGpuTest : public testing::Test {
 public:
  // Runs tests with |N| data.
  constexpr static size_t N = kThreadNum * 2;

  static void SetUpTestSuite() {
    GPU_MUST_SUCCESS(gpuDeviceReset(), "");
    xs_ = gpu::GpuMemory<bn254::G1AffinePointGpu>::MallocManaged(N);
    ys_ = gpu::GpuMemory<bn254::G1AffinePointGpu>::MallocManaged(N);
    affine_results_ = gpu::GpuMemory<bn254::G1AffinePointGpu>::MallocManaged(N);
    jacobian_results_ =
        gpu::GpuMemory<bn254::G1JacobianPointGpu>::MallocManaged(N);
    bool_results_ = gpu::GpuMemory<bool>::MallocManaged(N);

    bn254::G1Curve::Init();
    bn254::G1CurveGpu::Init();

    x_cpus_.reserve(N);
    y_cpus_.reserve(N);

    for (size_t i = 0; i < N; ++i) {
      bn254::G1AffinePoint x_cpu = bn254::G1AffinePoint::Random();
      bn254::G1AffinePoint y_cpu = bn254::G1AffinePoint::Random();

      xs_[i] = ConvertPoint<bn254::G1AffinePointGpu>(x_cpu);
      ys_[i] = ConvertPoint<bn254::G1AffinePointGpu>(y_cpu);

      x_cpus_.push_back(std::move(x_cpu));
      y_cpus_.push_back(std::move(y_cpu));
    }
  }

  static void TearDownTestSuite() {
    xs_.reset();
    ys_.reset();
    affine_results_.reset();
    jacobian_results_.reset();
    bool_results_.reset();

    GPU_MUST_SUCCESS(gpuDeviceReset(), "");

    x_cpus_.clear();
    y_cpus_.clear();
  }

  void SetUp() override {
    affine_results_.Memset();
    jacobian_results_.Memset();
    bool_results_.Memset();
  }

 protected:
  static gpu::GpuMemory<bn254::G1AffinePointGpu> xs_;
  static gpu::GpuMemory<bn254::G1AffinePointGpu> ys_;
  static gpu::GpuMemory<bn254::G1AffinePointGpu> affine_results_;
  static gpu::GpuMemory<bn254::G1JacobianPointGpu> jacobian_results_;
  static gpu::GpuMemory<bool> bool_results_;

  static std::vector<bn254::G1AffinePoint> x_cpus_;
  static std::vector<bn254::G1AffinePoint> y_cpus_;
};

gpu::GpuMemory<bn254::G1AffinePointGpu> AffinePointCorrectnessGpuTest::xs_;
gpu::GpuMemory<bn254::G1AffinePointGpu> AffinePointCorrectnessGpuTest::ys_;
gpu::GpuMemory<bn254::G1AffinePointGpu>
    AffinePointCorrectnessGpuTest::affine_results_;
gpu::GpuMemory<bn254::G1JacobianPointGpu>
    AffinePointCorrectnessGpuTest::jacobian_results_;
gpu::GpuMemory<bool> AffinePointCorrectnessGpuTest::bool_results_;

std::vector<bn254::G1AffinePoint> AffinePointCorrectnessGpuTest::x_cpus_;
std::vector<bn254::G1AffinePoint> AffinePointCorrectnessGpuTest::y_cpus_;

}  // namespace

TEST_F(AffinePointCorrectnessGpuTest, Add) {
  GPU_MUST_SUCCESS(LaunchAdd(xs_.get(), ys_.get(), jacobian_results_.get(), N),
                   "");
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(
        absl::Substitute("a: $0, b: $1", xs_[i].ToString(), ys_[i].ToString()));
    auto result = ConvertPoint<bn254::G1JacobianPoint>(jacobian_results_[i]);
    ASSERT_EQ(result, x_cpus_[i] + y_cpus_[i]);
  }
}

TEST_F(AffinePointCorrectnessGpuTest, Double) {
  GPU_MUST_SUCCESS(LaunchDouble(xs_.get(), jacobian_results_.get(), N), "");
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(absl::Substitute("a: $0", xs_[i].ToString()));
    auto result = ConvertPoint<bn254::G1JacobianPoint>(jacobian_results_[i]);
    ASSERT_EQ(result, x_cpus_[i].Double());
  }
}

TEST_F(AffinePointCorrectnessGpuTest, Negate) {
  GPU_MUST_SUCCESS(LaunchNegate(xs_.get(), affine_results_.get(), N), "");
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(absl::Substitute("a: $0", xs_[i].ToString()));
    auto result = ConvertPoint<bn254::G1AffinePoint>(affine_results_[i]);
    ASSERT_EQ(result, -x_cpus_[i]);
  }
}

TEST_F(AffinePointCorrectnessGpuTest, Eq) {
  GPU_MUST_SUCCESS(LaunchEq(xs_.get(), xs_.get(), bool_results_.get(), N), "");
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(
        absl::Substitute("a: $0, b: $1", xs_[i].ToString(), xs_[i].ToString()));
    ASSERT_TRUE(bool_results_[i]);
  }
}

TEST_F(AffinePointCorrectnessGpuTest, Ne) {
  GPU_MUST_SUCCESS(LaunchNe(xs_.get(), ys_.get(), bool_results_.get(), N), "");
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(
        absl::Substitute("a: $0, b: $1", xs_[i].ToString(), ys_[i].ToString()));
    ASSERT_TRUE(bool_results_[i]);
  }
}

}  // namespace tachyon::math
