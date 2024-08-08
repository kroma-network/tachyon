#include "absl/strings/substitute.h"
#include "gtest/gtest.h"

#include "tachyon/device/gpu/gpu_memory.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fq.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fq_gpu.h"
#include "tachyon/math/finite_fields/kernels/prime_field_ops.cu.h"
#include "tachyon/math/finite_fields/prime_field_conversions.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/test/launch_op_macros.h"

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

class PrimeFieldCorrectnessGpuTest : public FiniteFieldTest<bn254::Fq> {
 public:
  // Runs tests with |N| data.
  constexpr static size_t N = kThreadNum * 2;

  static void SetUpTestSuite() {
    FiniteFieldTest<bn254::Fq>::SetUpTestSuite();

    GPU_MUST_SUCCEED(gpuDeviceReset(), "");
    xs_ = gpu::GpuMemory<bn254::FqGpu>::MallocManaged(N);
    ys_ = gpu::GpuMemory<bn254::FqGpu>::MallocManaged(N);
    results_ = gpu::GpuMemory<bn254::FqGpu>::MallocManaged(N);

    x_cpus_.reserve(N);
    y_cpus_.reserve(N);

    for (size_t i = 0; i < N; ++i) {
      bn254::Fq x_cpu = bn254::Fq::Random();
      bn254::Fq y_cpu = bn254::Fq::Random();

      xs_[i] = ConvertPrimeField<bn254::FqGpu>(x_cpu);
      ys_[i] = ConvertPrimeField<bn254::FqGpu>(y_cpu);

      x_cpus_.push_back(std::move(x_cpu));
      y_cpus_.push_back(std::move(y_cpu));
    }
  }

  static void TearDownTestSuite() {
    xs_.reset();
    ys_.reset();
    results_.reset();

    GPU_MUST_SUCCEED(gpuDeviceReset(), "");

    x_cpus_.clear();
    y_cpus_.clear();
  }

  void SetUp() override { CHECK(results_.Memset()); }

 protected:
  static gpu::GpuMemory<bn254::FqGpu> xs_;
  static gpu::GpuMemory<bn254::FqGpu> ys_;
  static gpu::GpuMemory<bn254::FqGpu> results_;

  static std::vector<bn254::Fq> x_cpus_;
  static std::vector<bn254::Fq> y_cpus_;
};

gpu::GpuMemory<bn254::FqGpu> PrimeFieldCorrectnessGpuTest::xs_;
gpu::GpuMemory<bn254::FqGpu> PrimeFieldCorrectnessGpuTest::ys_;
gpu::GpuMemory<bn254::FqGpu> PrimeFieldCorrectnessGpuTest::results_;

std::vector<bn254::Fq> PrimeFieldCorrectnessGpuTest::x_cpus_;
std::vector<bn254::Fq> PrimeFieldCorrectnessGpuTest::y_cpus_;

}  // namespace

#define RUN_OPERATION_TESTS(method)                                            \
  GPU_MUST_SUCCEED(Launch##method(xs_.data(), ys_.data(), results_.data(), N), \
                   "");                                                        \
  for (size_t i = 0; i < N; ++i)

TEST_F(PrimeFieldCorrectnessGpuTest, Add) {
  RUN_OPERATION_TESTS(Add) {
    SCOPED_TRACE(
        absl::Substitute("a: $0, b: $1", xs_[i].ToString(), ys_[i].ToString()));
    ASSERT_EQ(ConvertPrimeField<bn254::Fq>(results_[i]),
              x_cpus_[i] + y_cpus_[i]);
  }
}

TEST_F(PrimeFieldCorrectnessGpuTest, Sub) {
  RUN_OPERATION_TESTS(Sub) {
    SCOPED_TRACE(
        absl::Substitute("a: $0, b: $1", xs_[i].ToString(), ys_[i].ToString()));
    ASSERT_EQ(ConvertPrimeField<bn254::Fq>(results_[i]),
              x_cpus_[i] - y_cpus_[i]);
  }
}

TEST_F(PrimeFieldCorrectnessGpuTest, Mul) {
  RUN_OPERATION_TESTS(Mul) {
    SCOPED_TRACE(
        absl::Substitute("a: $0, b: $1", xs_[i].ToString(), ys_[i].ToString()));
    ASSERT_EQ(ConvertPrimeField<bn254::Fq>(results_[i]),
              x_cpus_[i] * y_cpus_[i]);
  }
}

TEST_F(PrimeFieldCorrectnessGpuTest, Div) {
  RUN_OPERATION_TESTS(Div) {
    SCOPED_TRACE(
        absl::Substitute("a: $0, b: $1", xs_[i].ToString(), ys_[i].ToString()));
    ASSERT_EQ(ConvertPrimeField<bn254::Fq>(results_[i]),
              x_cpus_[i] / y_cpus_[i]);
  }
}

#undef RUN_OPERATION_TESTS

}  // namespace tachyon::math
