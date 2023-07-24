#include "absl/strings/substitute.h"
#include "gtest/gtest.h"

#include "tachyon/device/gpu/cuda/cuda_memory.h"
#include "tachyon/math/elliptic_curves/bn/bn254/curve_config_cuda.cu.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/kernels/elliptic_curve_ops.cu.h"
#include "tachyon/math/test/launch_op_macros.cu.h"

namespace tachyon {
namespace math {

namespace {

#define DEFINE_LAUNCH_FIELD_BINARY_OP(method)                   \
  DEFINE_LAUNCH_BINARY_OP(32, method, bn254::G1AffinePointCuda, \
                          bn254::G1JacobianPointCuda)

DEFINE_LAUNCH_FIELD_BINARY_OP(Add)

DEFINE_LAUNCH_UNARY_OP(32, Double, bn254::G1AffinePointCuda,
                       bn254::G1JacobianPointCuda)
DEFINE_LAUNCH_UNARY_OP(32, Negative, bn254::G1AffinePointCuda,
                       bn254::G1AffinePointCuda)

#undef DEFINE_LAUNCH_FIELD_BINARY_OP

#define DEFINE_LAUNCH_COMPARISON_OP(method) \
  DEFINE_LAUNCH_BINARY_OP(32, method, bn254::G1AffinePointCuda, bool)

DEFINE_LAUNCH_COMPARISON_OP(Eq)
DEFINE_LAUNCH_COMPARISON_OP(Ne)

#undef DEFINE_LAUNCH_COMPARISON_OP

class AffinePointCorrectnessCudaTest : public testing::Test {
 public:
  constexpr static size_t N = 1000;

  static void SetUpTestSuite() {
    GPU_SUCCESS(cudaDeviceReset());
    xs_ = device::gpu::MakeManagedUnique<bn254::G1AffinePointCuda>(
        N * sizeof(bn254::G1AffinePointCuda));
    ys_ = device::gpu::MakeManagedUnique<bn254::G1AffinePointCuda>(
        N * sizeof(bn254::G1AffinePointCuda));
    affine_results_ = device::gpu::MakeManagedUnique<bn254::G1AffinePointCuda>(
        N * sizeof(bn254::G1AffinePointCuda));
    jacobian_results_ =
        device::gpu::MakeManagedUnique<bn254::G1JacobianPointCuda>(
            N * sizeof(bn254::G1JacobianPointCuda));
    bool_results_ = device::gpu::MakeManagedUnique<bool>(N * sizeof(bool));

    bn254::CurveConfig<bn254::FqGmp, bn254::FrGmp>::Init();
    bn254::CurveConfig<bn254::FqCuda, bn254::FrCuda>::Init();

    x_gmps_.reserve(N);
    y_gmps_.reserve(N);

    for (size_t i = 0; i < N; ++i) {
      bn254::G1AffinePointGmp x_gmp = bn254::G1AffinePointGmp::Random();
      bn254::G1AffinePointGmp y_gmp = bn254::G1AffinePointGmp::Random();

      (xs_.get())[i] = bn254::G1AffinePointCuda(
          bn254::FqCuda::FromMontgomery(x_gmp.x().ToMontgomery()),
          bn254::FqCuda::FromMontgomery(x_gmp.y().ToMontgomery()));
      (ys_.get())[i] = bn254::G1AffinePointCuda(
          bn254::FqCuda::FromMontgomery(y_gmp.x().ToMontgomery()),
          bn254::FqCuda::FromMontgomery(y_gmp.y().ToMontgomery()));

      x_gmps_.push_back(std::move(x_gmp));
      y_gmps_.push_back(std::move(y_gmp));
    }
  }

  static void TearDownTestSuite() {
    xs_.reset();
    ys_.reset();
    affine_results_.reset();
    jacobian_results_.reset();
    bool_results_.reset();

    GPU_SUCCESS(cudaDeviceReset());

    x_gmps_.clear();
    y_gmps_.clear();
  }

  void SetUp() override {
    GPU_SUCCESS(cudaMemset(affine_results_.get(), 0,
                           N * sizeof(bn254::G1AffinePointCuda)));
    GPU_SUCCESS(cudaMemset(jacobian_results_.get(), 0,
                           N * sizeof(bn254::G1JacobianPointCuda)));
    GPU_SUCCESS(cudaMemset(bool_results_.get(), 0, N * sizeof(bool)));
  }

 protected:
  static device::gpu::ScopedMemory<bn254::G1AffinePointCuda> xs_;
  static device::gpu::ScopedMemory<bn254::G1AffinePointCuda> ys_;
  static device::gpu::ScopedMemory<bn254::G1AffinePointCuda> affine_results_;
  static device::gpu::ScopedMemory<bn254::G1JacobianPointCuda>
      jacobian_results_;
  static device::gpu::ScopedMemory<bool> bool_results_;

  static std::vector<bn254::G1AffinePointGmp> x_gmps_;
  static std::vector<bn254::G1AffinePointGmp> y_gmps_;
};

device::gpu::ScopedMemory<bn254::G1AffinePointCuda>
    AffinePointCorrectnessCudaTest::xs_;
device::gpu::ScopedMemory<bn254::G1AffinePointCuda>
    AffinePointCorrectnessCudaTest::ys_;
device::gpu::ScopedMemory<bn254::G1AffinePointCuda>
    AffinePointCorrectnessCudaTest::affine_results_;
device::gpu::ScopedMemory<bn254::G1JacobianPointCuda>
    AffinePointCorrectnessCudaTest::jacobian_results_;
device::gpu::ScopedMemory<bool> AffinePointCorrectnessCudaTest::bool_results_;

std::vector<bn254::G1AffinePointGmp> AffinePointCorrectnessCudaTest::x_gmps_;
std::vector<bn254::G1AffinePointGmp> AffinePointCorrectnessCudaTest::y_gmps_;

}  // namespace

TEST_F(AffinePointCorrectnessCudaTest, Add) {
  GPU_SUCCESS(LaunchAdd(xs_.get(), ys_.get(), jacobian_results_.get(), N));
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(absl::Substitute("a: $0, b: $1", (xs_.get())[i].ToString(),
                                  (ys_.get())[i].ToString()));
    auto result = bn254::G1JacobianPointGmp(
        bn254::FqGmp::FromMontgomery(
            (jacobian_results_.get())[i].x().ToMontgomery()),
        bn254::FqGmp::FromMontgomery(
            (jacobian_results_.get())[i].y().ToMontgomery()),
        bn254::FqGmp::FromMontgomery(
            (jacobian_results_.get())[i].z().ToMontgomery()));
    ASSERT_EQ(result, x_gmps_[i] + y_gmps_[i]);
  }
}

// TODO(chokobole): Enable this test.
// TEST_F(AffinePointCorrectnessCudaTest, Double) {
//   GPU_SUCCESS(LaunchDouble(xs_.get(), jacobian_results_.get(), N));
//   for (size_t i = 0; i < N; ++i) {
//     SCOPED_TRACE(absl::Substitute("a: $0", (xs_.get())[i].ToString()));
//     auto result = bn254::G1JacobianPointGmp(
//         bn254::FqGmp::FromMontgomery(
//             (jacobian_results_.get())[i].x().ToMontgomery()),
//         bn254::FqGmp::FromMontgomery(
//             (jacobian_results_.get())[i].y().ToMontgomery()),
//         bn254::FqGmp::FromMontgomery(
//             (jacobian_results_.get())[i].z().ToMontgomery()));
//     ASSERT_EQ(result, x_gmps_[i].Double());
//   }
// }

TEST_F(AffinePointCorrectnessCudaTest, Negative) {
  GPU_SUCCESS(LaunchNegative(xs_.get(), affine_results_.get(), N));
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(absl::Substitute("a: $0", (xs_.get())[i].ToString()));
    auto result = bn254::G1AffinePointGmp(
        bn254::FqGmp::FromMontgomery(
            (affine_results_.get())[i].x().ToMontgomery()),
        bn254::FqGmp::FromMontgomery(
            (affine_results_.get())[i].y().ToMontgomery()));
    ASSERT_EQ(result, x_gmps_[i].Negative());
  }
}

TEST_F(AffinePointCorrectnessCudaTest, Eq) {
  GPU_SUCCESS(LaunchEq(xs_.get(), xs_.get(), bool_results_.get(), N));
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(absl::Substitute("a: $0, b: $1", (xs_.get())[i].ToString(),
                                  (xs_.get())[i].ToString()));
    ASSERT_TRUE((bool_results_.get())[i]);
  }
}

TEST_F(AffinePointCorrectnessCudaTest, Ne) {
  GPU_SUCCESS(LaunchNe(xs_.get(), ys_.get(), bool_results_.get(), N));
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(absl::Substitute("a: $0, b: $1", (xs_.get())[i].ToString(),
                                  (ys_.get())[i].ToString()));
    ASSERT_TRUE((bool_results_.get())[i]);
  }
}

}  // namespace math
}  // namespace tachyon
