#include "absl/strings/substitute.h"
#include "gtest/gtest.h"

#include "tachyon/device/gpu/cuda/cuda_memory.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1_cuda.cu.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/kernels/elliptic_curve_ops.cu.h"
#include "tachyon/math/test/launch_op_macros.cu.h"

namespace tachyon {
namespace math {

namespace {

#define DEFINE_LAUNCH_FIELD_BINARY_OP(method)                       \
  DEFINE_LAUNCH_BINARY_OP(32, method, bn254::G1ProjectivePointCuda, \
                          bn254::G1ProjectivePointCuda)             \
  DEFINE_LAUNCH_BINARY_OP(32, method, bn254::G1JacobianPointCuda,   \
                          bn254::G1JacobianPointCuda)               \
  DEFINE_LAUNCH_BINARY_OP(32, method, bn254::G1PointXYZZCuda,       \
                          bn254::G1PointXYZZCuda)

DEFINE_LAUNCH_FIELD_BINARY_OP(Add)

#undef DEFINE_LAUNCH_FIELD_BINARY_OP

#define DEFINE_LAUNCH_FIELD_UNARY_OP(method)                       \
  DEFINE_LAUNCH_UNARY_OP(32, method, bn254::G1ProjectivePointCuda, \
                         bn254::G1ProjectivePointCuda)             \
  DEFINE_LAUNCH_UNARY_OP(32, method, bn254::G1JacobianPointCuda,   \
                         bn254::G1JacobianPointCuda)               \
  DEFINE_LAUNCH_UNARY_OP(32, method, bn254::G1PointXYZZCuda,       \
                         bn254::G1PointXYZZCuda)

DEFINE_LAUNCH_FIELD_UNARY_OP(Double)
DEFINE_LAUNCH_FIELD_UNARY_OP(Negative)

#undef DEFINE_LAUNCH_FIELD_UNARY_OP

#define DEFINE_LAUNCH_COMPARISON_OP(method)                               \
  DEFINE_LAUNCH_BINARY_OP(32, method, bn254::G1ProjectivePointCuda, bool) \
  DEFINE_LAUNCH_BINARY_OP(32, method, bn254::G1JacobianPointCuda, bool)   \
  DEFINE_LAUNCH_BINARY_OP(32, method, bn254::G1PointXYZZCuda, bool)

DEFINE_LAUNCH_COMPARISON_OP(Eq)
DEFINE_LAUNCH_COMPARISON_OP(Ne)

#undef DEFINE_LAUNCH_COMPARISON_OP

template <typename PointType>
class PointCorrectnessCudaTest : public testing::Test {
 public:
  using Actual = typename PointType::Actual;
  using Expected = typename PointType::Expected;

  constexpr static size_t N = 1000;

  static void SetUpTestSuite() {
    GPU_SUCCESS(cudaDeviceReset());
    xs_ = device::gpu::MakeManagedUnique<Actual>(N * sizeof(Actual));
    ys_ = device::gpu::MakeManagedUnique<Actual>(N * sizeof(Actual));
    results_ = device::gpu::MakeManagedUnique<Actual>(N * sizeof(Actual));
    bool_results_ = device::gpu::MakeManagedUnique<bool>(N * sizeof(bool));

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

    GPU_SUCCESS(cudaDeviceReset());

    x_gmps_.clear();
    y_gmps_.clear();
  }

  void SetUp() override {
    GPU_SUCCESS(cudaMemset(results_.get(), 0, N * sizeof(Actual)));
    GPU_SUCCESS(cudaMemset(bool_results_.get(), 0, N * sizeof(bool)));
  }

 protected:
  static device::gpu::ScopedMemory<Actual> xs_;
  static device::gpu::ScopedMemory<Actual> ys_;
  static device::gpu::ScopedMemory<Actual> results_;
  static device::gpu::ScopedMemory<bool> bool_results_;

  static std::vector<Expected> x_gmps_;
  static std::vector<Expected> y_gmps_;
};

template <typename PointType>
device::gpu::ScopedMemory<typename PointCorrectnessCudaTest<PointType>::Actual>
    PointCorrectnessCudaTest<PointType>::xs_;

template <typename PointType>
device::gpu::ScopedMemory<typename PointCorrectnessCudaTest<PointType>::Actual>
    PointCorrectnessCudaTest<PointType>::ys_;

template <typename PointType>
device::gpu::ScopedMemory<typename PointCorrectnessCudaTest<PointType>::Actual>
    PointCorrectnessCudaTest<PointType>::results_;

template <typename PointType>
device::gpu::ScopedMemory<bool>
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

  GPU_SUCCESS(
      LaunchAdd(this->xs_.get(), this->ys_.get(), this->results_.get(), N));
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

  GPU_SUCCESS(LaunchDouble(this->xs_.get(), this->results_.get(), N));
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

  GPU_SUCCESS(LaunchNegative(this->xs_.get(), this->results_.get(), N));
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(absl::Substitute("a: $0", (this->xs_.get())[i].ToString()));
    auto result =
        Expected::FromMontgomery((this->results_.get())[i].ToMontgomery());
    ASSERT_EQ(result, this->x_gmps_[i].Negative());
  }
}

TYPED_TEST(PointCorrectnessCudaTest, Eq) {
  size_t N = PointCorrectnessCudaTest<TypeParam>::N;

  GPU_SUCCESS(
      LaunchEq(this->xs_.get(), this->xs_.get(), this->bool_results_.get(), N));
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(absl::Substitute("a: $0, b: $1",
                                  (this->xs_.get())[i].ToString(),
                                  (this->xs_.get())[i].ToString()));
    ASSERT_TRUE((this->bool_results_.get())[i]);
  }
}

TYPED_TEST(PointCorrectnessCudaTest, Ne) {
  size_t N = PointCorrectnessCudaTest<TypeParam>::N;

  GPU_SUCCESS(
      LaunchNe(this->xs_.get(), this->ys_.get(), this->bool_results_.get(), N));
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(absl::Substitute("a: $0, b: $1",
                                  (this->xs_.get())[i].ToString(),
                                  (this->ys_.get())[i].ToString()));
    ASSERT_TRUE((this->bool_results_.get())[i]);
  }
}

}  // namespace math
}  // namespace tachyon
