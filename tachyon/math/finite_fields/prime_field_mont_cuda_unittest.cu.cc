#include "gtest/gtest.h"

#include "tachyon/device/gpu/cuda/cuda_memory.h"
#include "tachyon/math/finite_fields/kernels/prime_field_ops.cu.h"
#include "tachyon/math/finite_fields/kernels/test/launch_op_macros.cu.h"
#include "tachyon/math/finite_fields/prime_field.h"

namespace tachyon {
namespace math {

#if TACHYON_CUDA
using GF7Cuda = PrimeFieldMontCuda<GF7Config>;
#endif  // TACHYON_CUDA

namespace {

#define DEFINE_LAUNCH_FIELD_OP(method) \
  DEFINE_LAUNCH_OP(32, method, GF7Cuda, GF7Cuda)
#define DEFINE_LAUNCH_COMPARISON_OP(method) \
  DEFINE_LAUNCH_OP(32, method, GF7Cuda, bool)

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

class PrimeFieldMontCudaTest : public testing::Test {
 public:
  constexpr static size_t N = 5;

  static void SetUpTestSuite() {
    GPU_SUCCESS(cudaDeviceReset());
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

    GPU_SUCCESS(cudaDeviceReset());
  }

  void SetUp() override {
    GPU_SUCCESS(cudaMemset(xs_.get(), 0, N * sizeof(GF7Cuda)));
    GPU_SUCCESS(cudaMemset(ys_.get(), 0, N * sizeof(GF7Cuda)));
    GPU_SUCCESS(cudaMemset(results_.get(), 0, N * sizeof(GF7Cuda)));
    GPU_SUCCESS(cudaMemset(bool_results_.get(), 0, N * sizeof(bool)));
  }

 protected:
  static device::gpu::ScopedMemory<GF7Cuda> xs_;
  static device::gpu::ScopedMemory<GF7Cuda> ys_;
  static device::gpu::ScopedMemory<GF7Cuda> results_;
  static device::gpu::ScopedMemory<bool> bool_results_;
};

device::gpu::ScopedMemory<GF7Cuda> PrimeFieldMontCudaTest::xs_;
device::gpu::ScopedMemory<GF7Cuda> PrimeFieldMontCudaTest::ys_;
device::gpu::ScopedMemory<GF7Cuda> PrimeFieldMontCudaTest::results_;
device::gpu::ScopedMemory<bool> PrimeFieldMontCudaTest::bool_results_;

}  // namespace

#define RUN_OPERATION_TESTS(method, results)                                  \
  for (size_t i = 0; i < std::size(tests); ++i) {                             \
    const auto& test = tests[i];                                              \
    (xs_.get())[i] = GF7Cuda::FromBigInt(test.x.ToBigInt());                  \
    (ys_.get())[i] = GF7Cuda::FromBigInt(test.y.ToBigInt());                  \
  }                                                                           \
  GPU_SUCCESS(                                                                \
      Launch##method(xs_.get(), ys_.get(), results.get(), std::size(tests))); \
  for (size_t i = 0; i < std::size(tests); ++i)

#define RUN_FIELD_OPERATION_TESTS(method) \
  RUN_OPERATION_TESTS(method, results_)   \
  ASSERT_EQ(GF7::FromBigInt((results_.get())[i].ToBigInt()), tests[i].result)

#define RUN_COMPARISON_OPERATION_TESTS(method) \
  RUN_OPERATION_TESTS(method, bool_results_)   \
  ASSERT_EQ((bool_results_.get())[i], tests[i].result)

TEST_F(PrimeFieldMontCudaTest, FromString) {
  EXPECT_EQ(GF7Cuda::FromDecString("3"), GF7Cuda(3));
  EXPECT_EQ(GF7Cuda::FromHexString("0x3"), GF7Cuda(3));
}

TEST_F(PrimeFieldMontCudaTest, ToString) {
  GF7Cuda f(3);

  EXPECT_EQ(f.ToString(), "3");
  EXPECT_EQ(f.ToHexString(), "0x3");
}

TEST_F(PrimeFieldMontCudaTest, Zero) {
  EXPECT_TRUE(GF7Cuda::Zero().IsZero());
  EXPECT_FALSE(GF7Cuda::One().IsZero());
}

TEST_F(PrimeFieldMontCudaTest, One) {
  EXPECT_TRUE(GF7Cuda::One().IsOne());
  EXPECT_FALSE(GF7Cuda::Zero().IsOne());
}

TEST_F(PrimeFieldMontCudaTest, Add) {
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

TEST_F(PrimeFieldMontCudaTest, Sub) {
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

TEST_F(PrimeFieldMontCudaTest, Mul) {
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

TEST_F(PrimeFieldMontCudaTest, Eq) {
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

  GPU_SUCCESS(
      LaunchEq(xs_.get(), ys_.get(), bool_results_.get(), std::size(tests)));
  for (size_t i = 0; i < std::size(tests); ++i) {
    ASSERT_EQ((bool_results_.get())[i], tests[i].result);
  }
}

TEST_F(PrimeFieldMontCudaTest, Ne) {
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

  GPU_SUCCESS(
      LaunchNe(xs_.get(), ys_.get(), bool_results_.get(), std::size(tests)));
  for (size_t i = 0; i < std::size(tests); ++i) {
    ASSERT_EQ((bool_results_.get())[i], tests[i].result);
  }
}

TEST_F(PrimeFieldMontCudaTest, Lt) {
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

TEST_F(PrimeFieldMontCudaTest, Le) {
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

TEST_F(PrimeFieldMontCudaTest, Gt) {
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

TEST_F(PrimeFieldMontCudaTest, Ge) {
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

#undef RUN_COMPARISON_OPERATION_TESTS
#undef RUN_FIELD_OPERATION_TESTS
#undef RUN_OPERATION_TESTS

}  // namespace math
}  // namespace tachyon
