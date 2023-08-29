#include "absl/strings/substitute.h"
#include "gtest/gtest.h"

#include "tachyon/device/gpu/cuda/scoped_memory.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1_cuda.cu.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/kernels/elliptic_curve_ops.cu.h"
#include "tachyon/math/test/launch_op_macros.cu.h"

namespace tachyon::math {

namespace {

constexpr size_t kThreadNum = 32;

#define DEFINE_LAUNCH_FIELD_BINARY_OP(method)                           \
  DEFINE_LAUNCH_BINARY_OP(kThreadNum, method, bn254::G1AffinePointCuda, \
                          bn254::G1JacobianPointCuda)

DEFINE_LAUNCH_FIELD_BINARY_OP(Add)

DEFINE_LAUNCH_UNARY_OP(kThreadNum, Double, bn254::G1AffinePointCuda,
                       bn254::G1JacobianPointCuda)
DEFINE_LAUNCH_UNARY_OP(kThreadNum, Negative, bn254::G1AffinePointCuda,
                       bn254::G1AffinePointCuda)

#undef DEFINE_LAUNCH_FIELD_BINARY_OP

#define DEFINE_LAUNCH_COMPARISON_OP(method) \
  DEFINE_LAUNCH_BINARY_OP(kThreadNum, method, bn254::G1AffinePointCuda, bool)

DEFINE_LAUNCH_COMPARISON_OP(Eq)
DEFINE_LAUNCH_COMPARISON_OP(Ne)

#undef DEFINE_LAUNCH_COMPARISON_OP

using namespace device;

class AffinePointCorrectnessCudaTest : public testing::Test {
 public:
  // Runs tests with |N| data.
  constexpr static size_t N = kThreadNum * 2;

  static void SetUpTestSuite() {
    GPU_MUST_SUCCESS(gpuDeviceReset(), "");
    xs_ = gpu::MallocManaged<bn254::G1AffinePointCuda>(N);
    ys_ = gpu::MallocManaged<bn254::G1AffinePointCuda>(N);
    affine_results_ = gpu::MallocManaged<bn254::G1AffinePointCuda>(N);
    jacobian_results_ = gpu::MallocManaged<bn254::G1JacobianPointCuda>(N);
    bool_results_ = gpu::MallocManaged<bool>(N);

    bn254::G1AffinePointGmp::Curve::Init();
    bn254::G1AffinePointCuda::Curve::Init();

    x_gmps_.reserve(N);
    y_gmps_.reserve(N);

    for (size_t i = 0; i < N; ++i) {
      bn254::G1AffinePointGmp x_gmp = bn254::G1AffinePointGmp::Random();
      bn254::G1AffinePointGmp y_gmp = bn254::G1AffinePointGmp::Random();

      (xs_.get())[i] =
          bn254::G1AffinePointCuda::FromMontgomery(x_gmp.ToMontgomery());
      (ys_.get())[i] =
          bn254::G1AffinePointCuda::FromMontgomery(y_gmp.ToMontgomery());

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

    GPU_MUST_SUCCESS(gpuDeviceReset(), "");

    x_gmps_.clear();
    y_gmps_.clear();
  }

  void SetUp() override {
    GPU_MUST_SUCCESS(gpuMemset(affine_results_.get(), 0,
                               N * sizeof(bn254::G1AffinePointCuda)),
                     "");
    GPU_MUST_SUCCESS(gpuMemset(jacobian_results_.get(), 0,
                               N * sizeof(bn254::G1JacobianPointCuda)),
                     "");
    GPU_MUST_SUCCESS(gpuMemset(bool_results_.get(), 0, N * sizeof(bool)), "");
  }

 protected:
  static gpu::ScopedUnifiedMemory<bn254::G1AffinePointCuda> xs_;
  static gpu::ScopedUnifiedMemory<bn254::G1AffinePointCuda> ys_;
  static gpu::ScopedUnifiedMemory<bn254::G1AffinePointCuda> affine_results_;
  static gpu::ScopedUnifiedMemory<bn254::G1JacobianPointCuda> jacobian_results_;
  static gpu::ScopedUnifiedMemory<bool> bool_results_;

  static std::vector<bn254::G1AffinePointGmp> x_gmps_;
  static std::vector<bn254::G1AffinePointGmp> y_gmps_;
};

gpu::ScopedUnifiedMemory<bn254::G1AffinePointCuda>
    AffinePointCorrectnessCudaTest::xs_;
gpu::ScopedUnifiedMemory<bn254::G1AffinePointCuda>
    AffinePointCorrectnessCudaTest::ys_;
gpu::ScopedUnifiedMemory<bn254::G1AffinePointCuda>
    AffinePointCorrectnessCudaTest::affine_results_;
gpu::ScopedUnifiedMemory<bn254::G1JacobianPointCuda>
    AffinePointCorrectnessCudaTest::jacobian_results_;
gpu::ScopedUnifiedMemory<bool> AffinePointCorrectnessCudaTest::bool_results_;

std::vector<bn254::G1AffinePointGmp> AffinePointCorrectnessCudaTest::x_gmps_;
std::vector<bn254::G1AffinePointGmp> AffinePointCorrectnessCudaTest::y_gmps_;

}  // namespace

TEST_F(AffinePointCorrectnessCudaTest, Add) {
  GPU_MUST_SUCCESS(LaunchAdd(xs_.get(), ys_.get(), jacobian_results_.get(), N),
                   "");
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(absl::Substitute("a: $0, b: $1", (xs_.get())[i].ToString(),
                                  (ys_.get())[i].ToString()));
    auto result = bn254::G1JacobianPointGmp::FromMontgomery(
        (jacobian_results_.get())[i].ToMontgomery());
    ASSERT_EQ(result, x_gmps_[i] + y_gmps_[i]);
  }
}

TEST_F(AffinePointCorrectnessCudaTest, Double) {
  GPU_MUST_SUCCESS(LaunchDouble(xs_.get(), jacobian_results_.get(), N), "");
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(absl::Substitute("a: $0", (xs_.get())[i].ToString()));
    auto result = bn254::G1JacobianPointGmp::FromMontgomery(
        (jacobian_results_.get())[i].ToMontgomery());
    ASSERT_EQ(result, x_gmps_[i].Double());
  }
}

TEST_F(AffinePointCorrectnessCudaTest, Negative) {
  GPU_MUST_SUCCESS(LaunchNegative(xs_.get(), affine_results_.get(), N), "");
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(absl::Substitute("a: $0", (xs_.get())[i].ToString()));
    auto result = bn254::G1AffinePointGmp::FromMontgomery(
        (affine_results_.get())[i].ToMontgomery());
    ASSERT_EQ(result, x_gmps_[i].Negative());
  }
}

TEST_F(AffinePointCorrectnessCudaTest, Eq) {
  GPU_MUST_SUCCESS(LaunchEq(xs_.get(), xs_.get(), bool_results_.get(), N), "");
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(absl::Substitute("a: $0, b: $1", (xs_.get())[i].ToString(),
                                  (xs_.get())[i].ToString()));
    ASSERT_TRUE((bool_results_.get())[i]);
  }
}

TEST_F(AffinePointCorrectnessCudaTest, Ne) {
  GPU_MUST_SUCCESS(LaunchNe(xs_.get(), ys_.get(), bool_results_.get(), N), "");
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(absl::Substitute("a: $0, b: $1", (xs_.get())[i].ToString(),
                                  (ys_.get())[i].ToString()));
    ASSERT_TRUE((bool_results_.get())[i]);
  }
}

}  // namespace tachyon::math
