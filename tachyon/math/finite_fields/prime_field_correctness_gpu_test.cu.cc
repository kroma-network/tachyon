#include "absl/strings/substitute.h"
#include "gtest/gtest.h"

#include "tachyon/device/gpu/gpu_memory.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fq_gpu.h"
#include "tachyon/math/finite_fields/kernels/prime_field_ops.cu.h"
#include "tachyon/math/test/launch_op_macros.cu.h"

namespace tachyon::math {

namespace {

constexpr size_t kThreadNum = 32;

#define DEFINE_LAUNCH_FIELD_OP(method) \
  DEFINE_LAUNCH_BINARY_OP(kThreadNum, method, bn254::FqGpu, bn254::FqGpu)

DEFINE_LAUNCH_FIELD_OP(Add)
DEFINE_LAUNCH_FIELD_OP(Sub)
DEFINE_LAUNCH_FIELD_OP(Mul)
DEFINE_LAUNCH_FIELD_OP(Div)

#undef DEFINE_LAUNCH_FIELD_OP

using namespace device;

class PrimeFieldCorrectnessGpuTest : public testing::Test {
 public:
  // Runs tests with |N| data.
  constexpr static size_t N = kThreadNum * 2;

  static void SetUpTestSuite() {
    GPU_MUST_SUCCESS(gpuDeviceReset(), "");
    xs_ = gpu::GpuMemory<bn254::FqGpu>::MallocManaged(N);
    ys_ = gpu::GpuMemory<bn254::FqGpu>::MallocManaged(N);
    results_ = gpu::GpuMemory<bn254::FqGpu>::MallocManaged(N);

    bn254::FqGmp::Init();

    x_gmps_.reserve(N);
    y_gmps_.reserve(N);

    for (size_t i = 0; i < N; ++i) {
      bn254::FqGmp x_gmp = bn254::FqGmp::Random();
      bn254::FqGmp y_gmp = bn254::FqGmp::Random();

      xs_[i] = bn254::FqGpu::FromMontgomery(x_gmp.ToMontgomery());
      ys_[i] = bn254::FqGpu::FromMontgomery(y_gmp.ToMontgomery());

      x_gmps_.push_back(std::move(x_gmp));
      y_gmps_.push_back(std::move(y_gmp));
    }
  }

  static void TearDownTestSuite() {
    xs_.reset();
    ys_.reset();
    results_.reset();

    GPU_MUST_SUCCESS(gpuDeviceReset(), "");

    x_gmps_.clear();
    y_gmps_.clear();
  }

  void SetUp() override { CHECK(results_.Memset()); }

 protected:
  static gpu::GpuMemory<bn254::FqGpu> xs_;
  static gpu::GpuMemory<bn254::FqGpu> ys_;
  static gpu::GpuMemory<bn254::FqGpu> results_;

  static std::vector<bn254::FqGmp> x_gmps_;
  static std::vector<bn254::FqGmp> y_gmps_;
};

gpu::GpuMemory<bn254::FqGpu> PrimeFieldCorrectnessGpuTest::xs_;
gpu::GpuMemory<bn254::FqGpu> PrimeFieldCorrectnessGpuTest::ys_;
gpu::GpuMemory<bn254::FqGpu> PrimeFieldCorrectnessGpuTest::results_;

std::vector<bn254::FqGmp> PrimeFieldCorrectnessGpuTest::x_gmps_;
std::vector<bn254::FqGmp> PrimeFieldCorrectnessGpuTest::y_gmps_;

}  // namespace

#define RUN_OPERATION_TESTS(method)                                         \
  GPU_MUST_SUCCESS(Launch##method(xs_.get(), ys_.get(), results_.get(), N), \
                   "");                                                     \
  for (size_t i = 0; i < N; ++i)

TEST_F(PrimeFieldCorrectnessGpuTest, Add) {
  RUN_OPERATION_TESTS(Add) {
    SCOPED_TRACE(
        absl::Substitute("a: $0, b: $1", xs_[i].ToString(), ys_[i].ToString()));
    ASSERT_EQ(results_[i].ToBigIntHost(), (x_gmps_[i] + y_gmps_[i]).ToBigInt());
  }
}

TEST_F(PrimeFieldCorrectnessGpuTest, Sub) {
  RUN_OPERATION_TESTS(Sub) {
    SCOPED_TRACE(
        absl::Substitute("a: $0, b: $1", xs_[i].ToString(), ys_[i].ToString()));
    ASSERT_EQ(results_[i].ToBigIntHost(), (x_gmps_[i] - y_gmps_[i]).ToBigInt());
  }
}

TEST_F(PrimeFieldCorrectnessGpuTest, Mul) {
  RUN_OPERATION_TESTS(Mul) {
    SCOPED_TRACE(
        absl::Substitute("a: $0, b: $1", xs_[i].ToString(), ys_[i].ToString()));
    ASSERT_EQ(results_[i].ToBigIntHost(), (x_gmps_[i] * y_gmps_[i]).ToBigInt());
  }
}

TEST_F(PrimeFieldCorrectnessGpuTest, Div) {
  RUN_OPERATION_TESTS(Div) {
    SCOPED_TRACE(
        absl::Substitute("a: $0, b: $1", xs_[i].ToString(), ys_[i].ToString()));
    ASSERT_EQ(results_[i].ToBigIntHost(), (x_gmps_[i] / y_gmps_[i]).ToBigInt());
  }
}

#undef RUN_OPERATION_TESTS

}  // namespace tachyon::math
