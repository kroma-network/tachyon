#include "absl/strings/substitute.h"
#include "gtest/gtest.h"

#include "tachyon/device/gpu/gpu_memory.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1_gpu.h"
#include "tachyon/math/elliptic_curves/point_conversions.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/kernels/elliptic_curve_ops.cu.h"
#include "tachyon/math/test/launch_op_macros.h"

namespace tachyon::math {

namespace {

constexpr size_t kThreadNum = 32;

#define DEFINE_LAUNCH_FIELD_BINARY_OP(method)                              \
  DEFINE_LAUNCH_BINARY_OP(kThreadNum, method, bn254::G1ProjectivePointGpu, \
                          bn254::G1ProjectivePointGpu)                     \
  DEFINE_LAUNCH_BINARY_OP(kThreadNum, method, bn254::G1JacobianPointGpu,   \
                          bn254::G1JacobianPointGpu)                       \
  DEFINE_LAUNCH_BINARY_OP(kThreadNum, method, bn254::G1PointXYZZGpu,       \
                          bn254::G1PointXYZZGpu)

DEFINE_LAUNCH_FIELD_BINARY_OP(Add)

#undef DEFINE_LAUNCH_FIELD_BINARY_OP

#define DEFINE_LAUNCH_FIELD_UNARY_OP(method)                              \
  DEFINE_LAUNCH_UNARY_OP(kThreadNum, method, bn254::G1ProjectivePointGpu, \
                         bn254::G1ProjectivePointGpu)                     \
  DEFINE_LAUNCH_UNARY_OP(kThreadNum, method, bn254::G1JacobianPointGpu,   \
                         bn254::G1JacobianPointGpu)                       \
  DEFINE_LAUNCH_UNARY_OP(kThreadNum, method, bn254::G1PointXYZZGpu,       \
                         bn254::G1PointXYZZGpu)

DEFINE_LAUNCH_FIELD_UNARY_OP(Double)
DEFINE_LAUNCH_FIELD_UNARY_OP(Negative)

#undef DEFINE_LAUNCH_FIELD_UNARY_OP

#define DEFINE_LAUNCH_COMPARISON_OP(method)                                    \
  DEFINE_LAUNCH_BINARY_OP(kThreadNum, method, bn254::G1ProjectivePointGpu,     \
                          bool)                                                \
  DEFINE_LAUNCH_BINARY_OP(kThreadNum, method, bn254::G1JacobianPointGpu, bool) \
  DEFINE_LAUNCH_BINARY_OP(kThreadNum, method, bn254::G1PointXYZZGpu, bool)

DEFINE_LAUNCH_COMPARISON_OP(Eq)
DEFINE_LAUNCH_COMPARISON_OP(Ne)

#undef DEFINE_LAUNCH_COMPARISON_OP

using namespace device;

template <typename PointType>
class PointCorrectnessGpuTest : public testing::Test {
 public:
  using Actual = typename PointType::Actual;
  using Expected = typename PointType::Expected;

  // Runs tests with |N| data.
  constexpr static size_t N = kThreadNum * 2;

  static void SetUpTestSuite() {
    GPU_MUST_SUCCESS(gpuDeviceReset(), "");
    xs_ = gpu::GpuMemory<Actual>::MallocManaged(N);
    ys_ = gpu::GpuMemory<Actual>::MallocManaged(N);
    results_ = gpu::GpuMemory<Actual>::MallocManaged(N);
    bool_results_ = gpu::GpuMemory<bool>::MallocManaged(N);

    Expected::Curve::Init();
    Actual::Curve::Init();

    x_gmps_.reserve(N);
    y_gmps_.reserve(N);

    for (size_t i = 0; i < N; ++i) {
      Expected x_gmp = Expected::Random();
      Expected y_gmp = Expected::Random();

      xs_[i] = ConvertPoint<Actual>(x_gmp);
      ys_[i] = ConvertPoint<Actual>(y_gmp);

      x_gmps_.push_back(std::move(x_gmp));
      y_gmps_.push_back(std::move(y_gmp));
    }
  }

  static void TearDownTestSuite() {
    xs_.reset();
    ys_.reset();
    results_.reset();
    bool_results_.reset();

    GPU_MUST_SUCCESS(gpuDeviceReset(), "");

    x_gmps_.clear();
    y_gmps_.clear();
  }

  void SetUp() override {
    CHECK(results_.Memset());
    CHECK(bool_results_.Memset());
  }

 protected:
  static gpu::GpuMemory<Actual> xs_;
  static gpu::GpuMemory<Actual> ys_;
  static gpu::GpuMemory<Actual> results_;
  static gpu::GpuMemory<bool> bool_results_;

  static std::vector<Expected> x_gmps_;
  static std::vector<Expected> y_gmps_;
};

template <typename PointType>
gpu::GpuMemory<typename PointCorrectnessGpuTest<PointType>::Actual>
    PointCorrectnessGpuTest<PointType>::xs_;

template <typename PointType>
gpu::GpuMemory<typename PointCorrectnessGpuTest<PointType>::Actual>
    PointCorrectnessGpuTest<PointType>::ys_;

template <typename PointType>
gpu::GpuMemory<typename PointCorrectnessGpuTest<PointType>::Actual>
    PointCorrectnessGpuTest<PointType>::results_;

template <typename PointType>
gpu::GpuMemory<bool> PointCorrectnessGpuTest<PointType>::bool_results_;

template <typename PointType>
std::vector<typename PointCorrectnessGpuTest<PointType>::Expected>
    PointCorrectnessGpuTest<PointType>::x_gmps_;

template <typename PointType>
std::vector<typename PointCorrectnessGpuTest<PointType>::Expected>
    PointCorrectnessGpuTest<PointType>::y_gmps_;

}  // namespace

struct ProjectivePointTypes {
  using Actual = bn254::G1ProjectivePointGpu;
  using Expected = bn254::G1ProjectivePointGmp;
};

struct JacobianPointTypes {
  using Actual = bn254::G1JacobianPointGpu;
  using Expected = bn254::G1JacobianPointGmp;
};

struct PointXYZZTypes {
  using Actual = bn254::G1PointXYZZGpu;
  using Expected = bn254::G1PointXYZZGmp;
};

using PointTypes =
    testing::Types<ProjectivePointTypes, JacobianPointTypes, PointXYZZTypes>;
TYPED_TEST_SUITE(PointCorrectnessGpuTest, PointTypes);

TYPED_TEST(PointCorrectnessGpuTest, Add) {
  using Expected = typename TypeParam::Expected;
  size_t N = PointCorrectnessGpuTest<TypeParam>::N;

  GPU_MUST_SUCCESS(
      LaunchAdd(this->xs_.get(), this->ys_.get(), this->results_.get(), N), "");
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(absl::Substitute("a: $0, b: $1", this->xs_[i].ToString(),
                                  this->ys_[i].ToString()));
    auto result = ConvertPoint<Expected>(this->results_[i]);
    ASSERT_EQ(result, this->x_gmps_[i] + this->y_gmps_[i]);
  }
}

TYPED_TEST(PointCorrectnessGpuTest, Double) {
  using Expected = typename TypeParam::Expected;
  size_t N = PointCorrectnessGpuTest<TypeParam>::N;

  GPU_MUST_SUCCESS(LaunchDouble(this->xs_.get(), this->results_.get(), N), "");
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(absl::Substitute("a: $0", this->xs_[i].ToString()));
    auto result = ConvertPoint<Expected>(this->results_[i]);
    ASSERT_EQ(result, this->x_gmps_[i].Double());
  }
}

TYPED_TEST(PointCorrectnessGpuTest, Negative) {
  using Expected = typename TypeParam::Expected;
  size_t N = PointCorrectnessGpuTest<TypeParam>::N;

  GPU_MUST_SUCCESS(LaunchNegative(this->xs_.get(), this->results_.get(), N),
                   "");
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(absl::Substitute("a: $0", this->xs_[i].ToString()));
    auto result = ConvertPoint<Expected>(this->results_[i]);
    ASSERT_EQ(result, -this->x_gmps_[i]);
  }
}

TYPED_TEST(PointCorrectnessGpuTest, Eq) {
  size_t N = PointCorrectnessGpuTest<TypeParam>::N;

  GPU_MUST_SUCCESS(
      LaunchEq(this->xs_.get(), this->xs_.get(), this->bool_results_.get(), N),
      "");
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(absl::Substitute("a: $0, b: $1", this->xs_[i].ToString(),
                                  this->xs_[i].ToString()));
    ASSERT_TRUE(this->bool_results_[i]);
  }
}

TYPED_TEST(PointCorrectnessGpuTest, Ne) {
  size_t N = PointCorrectnessGpuTest<TypeParam>::N;

  GPU_MUST_SUCCESS(
      LaunchNe(this->xs_.get(), this->ys_.get(), this->bool_results_.get(), N),
      "");
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(absl::Substitute("a: $0, b: $1", this->xs_[i].ToString(),
                                  this->ys_[i].ToString()));
    ASSERT_TRUE(this->bool_results_[i]);
  }
}

}  // namespace tachyon::math
