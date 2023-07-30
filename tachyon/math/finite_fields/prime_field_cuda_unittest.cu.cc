#include "gtest/gtest.h"

#include "tachyon/device/gpu/cuda/cuda_memory.h"
#include "tachyon/math/finite_fields/kernels/prime_field_ops.cu.h"
#include "tachyon/math/finite_fields/prime_field.cu.h"
#include "tachyon/math/test/launch_op_macros.cu.h"

namespace tachyon::math {

namespace {

constexpr size_t kThreadNum = 32;

#define DEFINE_LAUNCH_FIELD_OP(method) \
  DEFINE_LAUNCH_BINARY_OP(kThreadNum, method, GF7Cuda, GF7Cuda)
#define DEFINE_LAUNCH_COMPARISON_OP(method) \
  DEFINE_LAUNCH_BINARY_OP(kThreadNum, method, GF7Cuda, bool)

DEFINE_LAUNCH_FIELD_OP(Add)
DEFINE_LAUNCH_FIELD_OP(Sub)
DEFINE_LAUNCH_FIELD_OP(Mul)

DEFINE_LAUNCH_COMPARISON_OP(Eq)
DEFINE_LAUNCH_COMPARISON_OP(Ne)
DEFINE_LAUNCH_COMPARISON_OP(Lt)
DEFINE_LAUNCH_COMPARISON_OP(Le)
DEFINE_LAUNCH_COMPARISON_OP(Gt)
DEFINE_LAUNCH_COMPARISON_OP(Ge)

#undef DEFINE_LAUNCH_COMPARISON_OP
#undef DEFINE_LAUNCH_FIELD_OP

class PrimeFieldCudaTest : public testing::Test {
 public:
  // Runs tests with |N| data.
  constexpr static size_t N = kThreadNum * 2;

  static void SetUpTestSuite() {
    cudaError_t error = cudaDeviceReset();
    GPU_CHECK(error == cudaSuccess, error);
    xs_ = device::gpu::MakeManagedUnique<GF7Cuda>(N * sizeof(GF7Cuda));
    ys_ = device::gpu::MakeManagedUnique<GF7Cuda>(N * sizeof(GF7Cuda));
    results_ = device::gpu::MakeManagedUnique<GF7Cuda>(N * sizeof(GF7Cuda));
    bool_results_ = device::gpu::MakeManagedUnique<bool>(N * sizeof(bool));

    GF7Config::Init();
  }

  static void TearDownTestSuite() {
    xs_.reset();
    ys_.reset();
    results_.reset();
    bool_results_.reset();

    cudaError_t error = cudaDeviceReset();
    GPU_CHECK(error == cudaSuccess, error);
  }

  void SetUp() override {
    cudaError_t error = cudaMemset(xs_.get(), 0, N * sizeof(GF7Cuda));
    GPU_CHECK(error == cudaSuccess, error);
    error = cudaMemset(ys_.get(), 0, N * sizeof(GF7Cuda));
    GPU_CHECK(error == cudaSuccess, error);
    error = cudaMemset(results_.get(), 0, N * sizeof(GF7Cuda));
    GPU_CHECK(error == cudaSuccess, error);
    error = cudaMemset(bool_results_.get(), 0, N * sizeof(bool));
    GPU_CHECK(error == cudaSuccess, error);
  }

 protected:
  static device::gpu::ScopedMemory<GF7Cuda> xs_;
  static device::gpu::ScopedMemory<GF7Cuda> ys_;
  static device::gpu::ScopedMemory<GF7Cuda> results_;
  static device::gpu::ScopedMemory<bool> bool_results_;
};

device::gpu::ScopedMemory<GF7Cuda> PrimeFieldCudaTest::xs_;
device::gpu::ScopedMemory<GF7Cuda> PrimeFieldCudaTest::ys_;
device::gpu::ScopedMemory<GF7Cuda> PrimeFieldCudaTest::results_;
device::gpu::ScopedMemory<bool> PrimeFieldCudaTest::bool_results_;

}  // namespace

#define RUN_OPERATION_TESTS(method, results)                                 \
  for (size_t i = 0; i < std::size(tests); ++i) {                            \
    const auto& test = tests[i];                                             \
    (xs_.get())[i] = GF7Cuda::FromBigInt(test.x.ToBigInt());                 \
    (ys_.get())[i] = GF7Cuda::FromBigInt(test.y.ToBigInt());                 \
  }                                                                          \
  cudaError_t error =                                                        \
      Launch##method(xs_.get(), ys_.get(), results.get(), std::size(tests)); \
  GPU_CHECK(error == cudaSuccess, error);                                    \
  << "Failed to " method "()";                                               \
  for (size_t i = 0; i < std::size(tests); ++i)

#define RUN_FIELD_OPERATION_TESTS(method) \
  RUN_OPERATION_TESTS(method, results_)   \
  ASSERT_EQ(GF7::FromBigInt((results_.get())[i].ToBigInt()), tests[i].result)

#define RUN_COMPARISON_OPERATION_TESTS(method) \
  RUN_OPERATION_TESTS(method, bool_results_)   \
  ASSERT_EQ((bool_results_.get())[i], tests[i].result)

TEST_F(PrimeFieldCudaTest, FromString) {
  EXPECT_EQ(GF7Cuda::FromDecString("3"), GF7Cuda(3));
  EXPECT_EQ(GF7Cuda::FromHexString("0x3"), GF7Cuda(3));
}

TEST_F(PrimeFieldCudaTest, ToString) {
  GF7Cuda f(3);

  EXPECT_EQ(f.ToString(), "3");
  EXPECT_EQ(f.ToHexString(), "0x3");
}

TEST_F(PrimeFieldCudaTest, Zero) {
  EXPECT_TRUE(GF7Cuda::Zero().IsZero());
  EXPECT_FALSE(GF7Cuda::One().IsZero());
}

TEST_F(PrimeFieldCudaTest, One) {
  EXPECT_TRUE(GF7Cuda::One().IsOne());
  EXPECT_FALSE(GF7Cuda::Zero().IsOne());
}

TEST_F(PrimeFieldCudaTest, Add) {
  struct {
    GF7 x;
    GF7 y;
    GF7 result;
  } tests[] = {
      {GF7(3), GF7(2), GF7(5)},
      {GF7(2), GF7(3), GF7(5)},
      {GF7(5), GF7(3), GF7(1)},
      {GF7(3), GF7(5), GF7(1)},
  };

  RUN_FIELD_OPERATION_TESTS(Add);
}

TEST_F(PrimeFieldCudaTest, Sub) {
  struct {
    GF7 x;
    GF7 y;
    GF7 result;
  } tests[] = {
      {GF7(3), GF7(2), GF7(1)},
      {GF7(2), GF7(3), GF7(6)},
      {GF7(5), GF7(3), GF7(2)},
      {GF7(3), GF7(5), GF7(5)},
  };

  RUN_FIELD_OPERATION_TESTS(Sub);
}

TEST_F(PrimeFieldCudaTest, Mul) {
  struct {
    GF7 x;
    GF7 y;
    GF7 result;
  } tests[] = {
      {GF7(3), GF7(2), GF7(6)},
      {GF7(2), GF7(3), GF7(6)},
      {GF7(5), GF7(3), GF7(1)},
      {GF7(3), GF7(5), GF7(1)},
  };

  RUN_FIELD_OPERATION_TESTS(Mul);
}

TEST_F(PrimeFieldCudaTest, Eq) {
  struct {
    GF7 x;
    GF7 y;
    bool result;
  } tests[] = {
      {GF7(3), GF7(2), false},
      {GF7(2), GF7(3), false},
      {GF7(3), GF7(3), true},
  };

  for (size_t i = 0; i < std::size(tests); ++i) {
    const auto& test = tests[i];
    (xs_.get())[i] = GF7Cuda::FromBigInt(test.x.ToBigInt());
    (ys_.get())[i] = GF7Cuda::FromBigInt(test.y.ToBigInt());
    ASSERT_EQ((xs_.get())[i] == (ys_.get())[i], test.result);
  }

  cudaError_t error =
      LaunchEq(xs_.get(), ys_.get(), bool_results_.get(), std::size(tests));
  GPU_CHECK(error == cudaSuccess, error);
  for (size_t i = 0; i < std::size(tests); ++i) {
    ASSERT_EQ((bool_results_.get())[i], tests[i].result);
  }
}

TEST_F(PrimeFieldCudaTest, Ne) {
  struct {
    GF7 x;
    GF7 y;
    bool result;
  } tests[] = {
      {GF7(3), GF7(2), true},
      {GF7(2), GF7(3), true},
      {GF7(3), GF7(3), false},
  };

  for (size_t i = 0; i < std::size(tests); ++i) {
    const auto& test = tests[i];
    (xs_.get())[i] = GF7Cuda::FromBigInt(test.x.ToBigInt());
    (ys_.get())[i] = GF7Cuda::FromBigInt(test.y.ToBigInt());
    ASSERT_EQ((xs_.get())[i] != (ys_.get())[i], test.result);
  }

  cudaError_t error =
      LaunchNe(xs_.get(), ys_.get(), bool_results_.get(), std::size(tests));
  GPU_CHECK(error == cudaSuccess, error);
  for (size_t i = 0; i < std::size(tests); ++i) {
    ASSERT_EQ((bool_results_.get())[i], tests[i].result);
  }
}

TEST_F(PrimeFieldCudaTest, Lt) {
  struct {
    GF7 x;
    GF7 y;
    bool result;
  } tests[] = {
      {GF7(3), GF7(2), false},
      {GF7(2), GF7(3), true},
      {GF7(3), GF7(3), false},
  };

  RUN_COMPARISON_OPERATION_TESTS(Lt);
}

TEST_F(PrimeFieldCudaTest, Le) {
  struct {
    GF7 x;
    GF7 y;
    bool result;
  } tests[] = {
      {GF7(3), GF7(2), false},
      {GF7(2), GF7(3), true},
      {GF7(3), GF7(3), true},
  };

  RUN_COMPARISON_OPERATION_TESTS(Le);
}

TEST_F(PrimeFieldCudaTest, Gt) {
  struct {
    GF7 x;
    GF7 y;
    bool result;
  } tests[] = {
      {GF7(3), GF7(2), true},
      {GF7(2), GF7(3), false},
      {GF7(3), GF7(3), false},
  };

  RUN_COMPARISON_OPERATION_TESTS(Gt);
}

TEST_F(PrimeFieldCudaTest, Ge) {
  struct {
    GF7 x;
    GF7 y;
    bool result;
  } tests[] = {
      {GF7(3), GF7(2), true},
      {GF7(2), GF7(3), false},
      {GF7(3), GF7(3), true},
  };

  RUN_COMPARISON_OPERATION_TESTS(Ge);
}

#undef RUN_OPERATION_TESTS
#undef RUN_COMPARISON_OPERATION_TESTS
#undef RUN_FIELD_OPERATION_TESTS

}  // namespace tachyon::math
