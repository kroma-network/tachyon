#include "absl/strings/substitute.h"
#include "gtest/gtest.h"

#include "tachyon/device/gpu/cuda/cuda_memory.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fq_cuda.cu.h"
#include "tachyon/math/finite_fields/kernels/prime_field_ops.cu.h"
#include "tachyon/math/test/launch_op_macros.cu.h"

namespace tachyon::math {

namespace {

#define DEFINE_LAUNCH_FIELD_OP(method) \
  DEFINE_LAUNCH_BINARY_OP(32, method, bn254::FqCuda, bn254::FqCuda)

DEFINE_LAUNCH_FIELD_OP(Add)
DEFINE_LAUNCH_FIELD_OP(Sub)
DEFINE_LAUNCH_FIELD_OP(Mul)
DEFINE_LAUNCH_FIELD_OP(Div)

#undef DEFINE_LAUNCH_FIELD_OP

class PrimeFieldCorrectnessCudaTest : public testing::Test {
 public:
  constexpr static size_t N = 1000;

  static void SetUpTestSuite() {
    GPU_SUCCESS(cudaDeviceReset());
    xs_ = device::gpu::MakeManagedUnique<bn254::FqCuda>(N *
                                                        sizeof(bn254::FqCuda));
    ys_ = device::gpu::MakeManagedUnique<bn254::FqCuda>(N *
                                                        sizeof(bn254::FqCuda));
    results_ = device::gpu::MakeManagedUnique<bn254::FqCuda>(
        N * sizeof(bn254::FqCuda));

    bn254::FqGmp::Init();

    x_gmps_.reserve(N);
    y_gmps_.reserve(N);

    for (size_t i = 0; i < N; ++i) {
      bn254::FqGmp x_gmp = bn254::FqGmp::Random();
      bn254::FqGmp y_gmp = bn254::FqGmp::Random();

      (xs_.get())[i] = bn254::FqCuda::FromMontgomery(x_gmp.ToMontgomery());
      (ys_.get())[i] = bn254::FqCuda::FromMontgomery(y_gmp.ToMontgomery());

      x_gmps_.push_back(std::move(x_gmp));
      y_gmps_.push_back(std::move(y_gmp));
    }
  }

  static void TearDownTestSuite() {
    xs_.reset();
    ys_.reset();
    results_.reset();

    GPU_SUCCESS(cudaDeviceReset());

    x_gmps_.clear();
    y_gmps_.clear();
  }

  void SetUp() override {
    GPU_SUCCESS(cudaMemset(results_.get(), 0, N * sizeof(bn254::FqCuda)));
  }

 protected:
  static device::gpu::ScopedMemory<bn254::FqCuda> xs_;
  static device::gpu::ScopedMemory<bn254::FqCuda> ys_;
  static device::gpu::ScopedMemory<bn254::FqCuda> results_;

  static std::vector<bn254::FqGmp> x_gmps_;
  static std::vector<bn254::FqGmp> y_gmps_;
};

device::gpu::ScopedMemory<bn254::FqCuda> PrimeFieldCorrectnessCudaTest::xs_;
device::gpu::ScopedMemory<bn254::FqCuda> PrimeFieldCorrectnessCudaTest::ys_;
device::gpu::ScopedMemory<bn254::FqCuda>
    PrimeFieldCorrectnessCudaTest::results_;

std::vector<bn254::FqGmp> PrimeFieldCorrectnessCudaTest::x_gmps_;
std::vector<bn254::FqGmp> PrimeFieldCorrectnessCudaTest::y_gmps_;

}  // namespace

#define RUN_OPERATION_TESTS(method)                                     \
  GPU_SUCCESS(Launch##method(xs_.get(), ys_.get(), results_.get(), N)); \
  for (size_t i = 0; i < N; ++i)

TEST_F(PrimeFieldCorrectnessCudaTest, Add) {
  RUN_OPERATION_TESTS(Add) {
    SCOPED_TRACE(absl::Substitute("a: $0, b: $1", (xs_.get())[i].ToString(),
                                  (ys_.get())[i].ToString()));
    ASSERT_EQ((results_.get())[i].ToBigInt(),
              (x_gmps_[i] + y_gmps_[i]).ToBigInt());
  }
}

TEST_F(PrimeFieldCorrectnessCudaTest, Sub) {
  RUN_OPERATION_TESTS(Sub) {
    SCOPED_TRACE(absl::Substitute("a: $0, b: $1", (xs_.get())[i].ToString(),
                                  (ys_.get())[i].ToString()));
    ASSERT_EQ((results_.get())[i].ToBigInt(),
              (x_gmps_[i] - y_gmps_[i]).ToBigInt());
  }
}

TEST_F(PrimeFieldCorrectnessCudaTest, Mul) {
  RUN_OPERATION_TESTS(Mul) {
    SCOPED_TRACE(absl::Substitute("a: $0, b: $1", (xs_.get())[i].ToString(),
                                  (ys_.get())[i].ToString()));
    ASSERT_EQ((results_.get())[i].ToBigInt(),
              (x_gmps_[i] * y_gmps_[i]).ToBigInt());
  }
}

TEST_F(PrimeFieldCorrectnessCudaTest, Div) {
  RUN_OPERATION_TESTS(Div) {
    SCOPED_TRACE(absl::Substitute("a: $0, b: $1", (xs_.get())[i].ToString(),
                                  (ys_.get())[i].ToString()));
    ASSERT_EQ((results_.get())[i].ToBigInt(),
              (x_gmps_[i] / y_gmps_[i]).ToBigInt());
  }
}

#undef RUN_OPERATION_TESTS

}  // namespace tachyon::math
