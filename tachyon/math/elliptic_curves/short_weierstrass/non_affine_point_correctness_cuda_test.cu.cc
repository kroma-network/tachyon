#include "absl/strings/substitute.h"
#include "gtest/gtest.h"

#include "tachyon/device/gpu/cuda/scoped_memory.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1_cuda.cu.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/kernels/elliptic_curve_ops.cu.h"
#include "tachyon/math/test/launch_op_macros.cu.h"

namespace tachyon::math {

namespace {

constexpr size_t kThreadNum = 32;

#define DEFINE_LAUNCH_FIELD_BINARY_OP(method)                               \
  DEFINE_LAUNCH_BINARY_OP(kThreadNum, method, bn254::G1ProjectivePointCuda, \
                          bn254::G1ProjectivePointCuda)                     \
  DEFINE_LAUNCH_BINARY_OP(kThreadNum, method, bn254::G1JacobianPointCuda,   \
                          bn254::G1JacobianPointCuda)                       \
  DEFINE_LAUNCH_BINARY_OP(kThreadNum, method, bn254::G1PointXYZZCuda,       \
                          bn254::G1PointXYZZCuda)

DEFINE_LAUNCH_FIELD_BINARY_OP(Add)

#undef DEFINE_LAUNCH_FIELD_BINARY_OP

#define DEFINE_LAUNCH_FIELD_UNARY_OP(method)                               \
  DEFINE_LAUNCH_UNARY_OP(kThreadNum, method, bn254::G1ProjectivePointCuda, \
                         bn254::G1ProjectivePointCuda)                     \
  DEFINE_LAUNCH_UNARY_OP(kThreadNum, method, bn254::G1JacobianPointCuda,   \
                         bn254::G1JacobianPointCuda)                       \
  DEFINE_LAUNCH_UNARY_OP(kThreadNum, method, bn254::G1PointXYZZCuda,       \
                         bn254::G1PointXYZZCuda)

DEFINE_LAUNCH_FIELD_UNARY_OP(Double)
DEFINE_LAUNCH_FIELD_UNARY_OP(Negative)

#undef DEFINE_LAUNCH_FIELD_UNARY_OP

#define DEFINE_LAUNCH_COMPARISON_OP(method)                                 \
  DEFINE_LAUNCH_BINARY_OP(kThreadNum, method, bn254::G1ProjectivePointCuda, \
                          bool)                                             \
  DEFINE_LAUNCH_BINARY_OP(kThreadNum, method, bn254::G1JacobianPointCuda,   \
                          bool)                                             \
  DEFINE_LAUNCH_BINARY_OP(kThreadNum, method, bn254::G1PointXYZZCuda, bool)

DEFINE_LAUNCH_COMPARISON_OP(Eq)
DEFINE_LAUNCH_COMPARISON_OP(Ne)

#undef DEFINE_LAUNCH_COMPARISON_OP

using namespace device;

template <typename PointType>
class PointCorrectnessCudaTest : public testing::Test {
 public:
  using Actual = typename PointType::Actual;
  using Expected = typename PointType::Expected;

  // Runs tests with |N| data.
  constexpr static size_t N = kThreadNum * 2;

  static void SetUpTestSuite() {
    GPU_MUST_SUCCESS(gpuDeviceReset(), "");
    xs_ = gpu::MallocManaged<Actual>(N);
    ys_ = gpu::MallocManaged<Actual>(N);
    results_ = gpu::MallocManaged<Actual>(N);
    bool_results_ = gpu::MallocManaged<bool>(N);

    Expected::Curve::Init();
    Actual::Curve::Init();

    x_gmps_.reserve(N);
    y_gmps_.reserve(N);

    for (size_t i = 0; i < N; ++i) {
      Expected x_gmp = Expected::Random();
      Expected y_gmp = Expected::Random();

      (xs_.get())[i] = Actual::FromMontgomery(x_gmp.ToMontgomery());
      (ys_.get())[i] = Actual::FromMontgomery(y_gmp.ToMontgomery());

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
    GPU_MUST_SUCCESS(gpuMemset(results_.get(), 0, N * sizeof(Actual)), "");
    GPU_MUST_SUCCESS(gpuMemset(bool_results_.get(), 0, N * sizeof(bool)), "");
  }

 protected:
  static gpu::ScopedUnifiedMemory<Actual> xs_;
  static gpu::ScopedUnifiedMemory<Actual> ys_;
  static gpu::ScopedUnifiedMemory<Actual> results_;
  static gpu::ScopedUnifiedMemory<bool> bool_results_;

  static std::vector<Expected> x_gmps_;
  static std::vector<Expected> y_gmps_;
};

template <typename PointType>
gpu::ScopedUnifiedMemory<typename PointCorrectnessCudaTest<PointType>::Actual>
    PointCorrectnessCudaTest<PointType>::xs_;

template <typename PointType>
gpu::ScopedUnifiedMemory<typename PointCorrectnessCudaTest<PointType>::Actual>
    PointCorrectnessCudaTest<PointType>::ys_;

template <typename PointType>
gpu::ScopedUnifiedMemory<typename PointCorrectnessCudaTest<PointType>::Actual>
    PointCorrectnessCudaTest<PointType>::results_;

template <typename PointType>
gpu::ScopedUnifiedMemory<bool>
    PointCorrectnessCudaTest<PointType>::bool_results_;

template <typename PointType>
std::vector<typename PointCorrectnessCudaTest<PointType>::Expected>
    PointCorrectnessCudaTest<PointType>::x_gmps_;

template <typename PointType>
std::vector<typename PointCorrectnessCudaTest<PointType>::Expected>
    PointCorrectnessCudaTest<PointType>::y_gmps_;

}  // namespace

struct ProjectivePointTypes {
  using Actual = bn254::G1ProjectivePointCuda;
  using Expected = bn254::G1ProjectivePointGmp;
};

struct JacobianPointTypes {
  using Actual = bn254::G1JacobianPointCuda;
  using Expected = bn254::G1JacobianPointGmp;
};

struct PointXYZZTypes {
  using Actual = bn254::G1PointXYZZCuda;
  using Expected = bn254::G1PointXYZZGmp;
};

using PointTypes =
    testing::Types<ProjectivePointTypes, JacobianPointTypes, PointXYZZTypes>;
TYPED_TEST_SUITE(PointCorrectnessCudaTest, PointTypes);

TYPED_TEST(PointCorrectnessCudaTest, Add) {
  using Expected = typename TypeParam::Expected;
  size_t N = PointCorrectnessCudaTest<TypeParam>::N;

  GPU_MUST_SUCCESS(
      LaunchAdd(this->xs_.get(), this->ys_.get(), this->results_.get(), N), "");
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(absl::Substitute("a: $0, b: $1",
                                  (this->xs_.get())[i].ToString(),
                                  (this->ys_.get())[i].ToString()));
    auto result =
        Expected::FromMontgomery((this->results_.get())[i].ToMontgomery());
    ASSERT_EQ(result, this->x_gmps_[i] + this->y_gmps_[i]);
  }
}

TYPED_TEST(PointCorrectnessCudaTest, Double) {
  using Expected = typename TypeParam::Expected;
  size_t N = PointCorrectnessCudaTest<TypeParam>::N;

  GPU_MUST_SUCCESS(LaunchDouble(this->xs_.get(), this->results_.get(), N), "");
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(absl::Substitute("a: $0", (this->xs_.get())[i].ToString()));
    auto result =
        Expected::FromMontgomery((this->results_.get())[i].ToMontgomery());
    ASSERT_EQ(result, this->x_gmps_[i].Double());
  }
}

TYPED_TEST(PointCorrectnessCudaTest, Negative) {
  using Expected = typename TypeParam::Expected;
  size_t N = PointCorrectnessCudaTest<TypeParam>::N;

  GPU_MUST_SUCCESS(LaunchNegative(this->xs_.get(), this->results_.get(), N),
                   "");
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(absl::Substitute("a: $0", (this->xs_.get())[i].ToString()));
    auto result =
        Expected::FromMontgomery((this->results_.get())[i].ToMontgomery());
    ASSERT_EQ(result, this->x_gmps_[i].Negative());
  }
}

TYPED_TEST(PointCorrectnessCudaTest, Eq) {
  size_t N = PointCorrectnessCudaTest<TypeParam>::N;

  GPU_MUST_SUCCESS(
      LaunchEq(this->xs_.get(), this->xs_.get(), this->bool_results_.get(), N),
      "");
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(absl::Substitute("a: $0, b: $1",
                                  (this->xs_.get())[i].ToString(),
                                  (this->xs_.get())[i].ToString()));
    ASSERT_TRUE((this->bool_results_.get())[i]);
  }
}

TYPED_TEST(PointCorrectnessCudaTest, Ne) {
  size_t N = PointCorrectnessCudaTest<TypeParam>::N;

  GPU_MUST_SUCCESS(
      LaunchNe(this->xs_.get(), this->ys_.get(), this->bool_results_.get(), N),
      "");
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(absl::Substitute("a: $0, b: $1",
                                  (this->xs_.get())[i].ToString(),
                                  (this->ys_.get())[i].ToString()));
    ASSERT_TRUE((this->bool_results_.get())[i]);
  }
}

}  // namespace tachyon::math
