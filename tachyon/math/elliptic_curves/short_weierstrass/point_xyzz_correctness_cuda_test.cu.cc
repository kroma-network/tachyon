#include "absl/strings/substitute.h"
#include "gtest/gtest.h"

#include "tachyon/device/gpu/cuda/cuda_memory.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1_cuda.cu.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/kernels/elliptic_curve_ops.cu.h"
#include "tachyon/math/test/launch_op_macros.cu.h"

namespace tachyon {
namespace math {

namespace {

#define DEFINE_LAUNCH_FIELD_BINARY_OP(method)                 \
  DEFINE_LAUNCH_BINARY_OP(32, method, bn254::G1PointXYZZCuda, \
                          bn254::G1PointXYZZCuda)

DEFINE_LAUNCH_FIELD_BINARY_OP(Add)

#undef DEFINE_LAUNCH_FIELD_BINARY_OP

#define DEFINE_LAUNCH_FIELD_UNARY_OP(method)                 \
  DEFINE_LAUNCH_UNARY_OP(32, method, bn254::G1PointXYZZCuda, \
                         bn254::G1PointXYZZCuda)

DEFINE_LAUNCH_FIELD_UNARY_OP(Double)
DEFINE_LAUNCH_FIELD_UNARY_OP(Negative)

#undef DEFINE_LAUNCH_FIELD_UNARY_OP

#define DEFINE_LAUNCH_COMPARISON_OP(method) \
  DEFINE_LAUNCH_BINARY_OP(32, method, bn254::G1PointXYZZCuda, bool)

DEFINE_LAUNCH_COMPARISON_OP(Eq)
DEFINE_LAUNCH_COMPARISON_OP(Ne)

#undef DEFINE_LAUNCH_COMPARISON_OP

class PointXYZZCorrectnessCudaTest : public testing::Test {
 public:
  constexpr static size_t N = 1000;

  static void SetUpTestSuite() {
    GPU_SUCCESS(cudaDeviceReset());
    xs_ = device::gpu::MakeManagedUnique<bn254::G1PointXYZZCuda>(
        N * sizeof(bn254::G1PointXYZZCuda));
    ys_ = device::gpu::MakeManagedUnique<bn254::G1PointXYZZCuda>(
        N * sizeof(bn254::G1PointXYZZCuda));
    results_ = device::gpu::MakeManagedUnique<bn254::G1PointXYZZCuda>(
        N * sizeof(bn254::G1PointXYZZCuda));
    bool_results_ = device::gpu::MakeManagedUnique<bool>(N * sizeof(bool));

    bn254::G1AffinePointGmp::Curve::Init();
    bn254::G1AffinePointCuda::Curve::Init();

    x_gmps_.reserve(N);
    y_gmps_.reserve(N);

    for (size_t i = 0; i < N; ++i) {
      bn254::G1PointXYZZGmp x_gmp = bn254::G1PointXYZZGmp::Random();
      bn254::G1PointXYZZGmp y_gmp = bn254::G1PointXYZZGmp::Random();

      (xs_.get())[i] =
          bn254::G1PointXYZZCuda::FromMontgomery(x_gmp.ToMontgomery());
      (ys_.get())[i] =
          bn254::G1PointXYZZCuda::FromMontgomery(y_gmp.ToMontgomery());

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
    GPU_SUCCESS(
        cudaMemset(results_.get(), 0, N * sizeof(bn254::G1PointXYZZCuda)));
    GPU_SUCCESS(cudaMemset(bool_results_.get(), 0, N * sizeof(bool)));
  }

 protected:
  static device::gpu::ScopedMemory<bn254::G1PointXYZZCuda> xs_;
  static device::gpu::ScopedMemory<bn254::G1PointXYZZCuda> ys_;
  static device::gpu::ScopedMemory<bn254::G1PointXYZZCuda> results_;
  static device::gpu::ScopedMemory<bool> bool_results_;

  static std::vector<bn254::G1PointXYZZGmp> x_gmps_;
  static std::vector<bn254::G1PointXYZZGmp> y_gmps_;
};

device::gpu::ScopedMemory<bn254::G1PointXYZZCuda>
    PointXYZZCorrectnessCudaTest::xs_;
device::gpu::ScopedMemory<bn254::G1PointXYZZCuda>
    PointXYZZCorrectnessCudaTest::ys_;
device::gpu::ScopedMemory<bn254::G1PointXYZZCuda>
    PointXYZZCorrectnessCudaTest::results_;
device::gpu::ScopedMemory<bool> PointXYZZCorrectnessCudaTest::bool_results_;

std::vector<bn254::G1PointXYZZGmp> PointXYZZCorrectnessCudaTest::x_gmps_;
std::vector<bn254::G1PointXYZZGmp> PointXYZZCorrectnessCudaTest::y_gmps_;

}  // namespace

TEST_F(PointXYZZCorrectnessCudaTest, Add) {
  GPU_SUCCESS(LaunchAdd(xs_.get(), ys_.get(), results_.get(), N));
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(absl::Substitute("a: $0, b: $1", (xs_.get())[i].ToString(),
                                  (ys_.get())[i].ToString()));
    auto result = bn254::G1PointXYZZGmp::FromMontgomery(
        (results_.get())[i].ToMontgomery());
    ASSERT_EQ(result, x_gmps_[i] + y_gmps_[i]);
  }
}

TEST_F(PointXYZZCorrectnessCudaTest, Double) {
  GPU_SUCCESS(LaunchDouble(xs_.get(), results_.get(), N));
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(absl::Substitute("a: $0", (xs_.get())[i].ToString()));
    auto result = bn254::G1PointXYZZGmp::FromMontgomery(
        (results_.get())[i].ToMontgomery());
    ASSERT_EQ(result, x_gmps_[i].Double());
  }
}

TEST_F(PointXYZZCorrectnessCudaTest, Negative) {
  GPU_SUCCESS(LaunchNegative(xs_.get(), results_.get(), N));
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(absl::Substitute("a: $0", (xs_.get())[i].ToString()));
    auto result = bn254::G1PointXYZZGmp::FromMontgomery(
        (results_.get())[i].ToMontgomery());
    ASSERT_EQ(result, x_gmps_[i].Negative());
  }
}

TEST_F(PointXYZZCorrectnessCudaTest, Eq) {
  GPU_SUCCESS(LaunchEq(xs_.get(), xs_.get(), bool_results_.get(), N));
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(absl::Substitute("a: $0, b: $1", (xs_.get())[i].ToString(),
                                  (xs_.get())[i].ToString()));
    ASSERT_TRUE((bool_results_.get())[i]);
  }
}

TEST_F(PointXYZZCorrectnessCudaTest, Ne) {
  GPU_SUCCESS(LaunchNe(xs_.get(), ys_.get(), bool_results_.get(), N));
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(absl::Substitute("a: $0, b: $1", (xs_.get())[i].ToString(),
                                  (ys_.get())[i].ToString()));
    ASSERT_TRUE((bool_results_.get())[i]);
  }
}

}  // namespace math
}  // namespace tachyon
