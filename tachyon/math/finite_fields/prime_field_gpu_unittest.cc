#include "gtest/gtest.h"

#include "tachyon/device/gpu/gpu_memory.h"
#include "tachyon/math/finite_fields/kernels/prime_field_ops.cu.h"
#include "tachyon/math/finite_fields/prime_field_conversions.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/math/finite_fields/test/gf7_gpu.h"
#include "tachyon/math/test/launch_op_macros.h"

namespace tachyon::math {

namespace {

constexpr size_t kThreadNum = 32;

#define DEFINE_LAUNCH_FIELD_OP(method) \
  DEFINE_LAUNCH_BINARY_OP(kThreadNum, method, GF7Gpu, GF7Gpu)
#define DEFINE_LAUNCH_COMPARISON_OP(method) \
  DEFINE_LAUNCH_BINARY_OP(kThreadNum, method, GF7Gpu, bool)

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

using namespace device;

class PrimeFieldGpuTest : public FiniteFieldTest<GF7> {
 public:
  // Runs tests with |N| data.
  constexpr static size_t N = kThreadNum * 2;

  static void SetUpTestSuite() {
    FiniteFieldTest<GF7>::SetUpTestSuite();

    GPU_MUST_SUCCESS(gpuDeviceReset(), "");
    xs_ = gpu::GpuMemory<GF7Gpu>::MallocManaged(N);
    ys_ = gpu::GpuMemory<GF7Gpu>::MallocManaged(N);
    results_ = gpu::GpuMemory<GF7Gpu>::MallocManaged(N);
    bool_results_ = gpu::GpuMemory<bool>::MallocManaged(N);
  }

  static void TearDownTestSuite() {
    xs_.reset();
    ys_.reset();
    results_.reset();
    bool_results_.reset();

    GPU_MUST_SUCCESS(gpuDeviceReset(), "");
  }

  void SetUp() override {
    CHECK(xs_.Memset());
    CHECK(ys_.Memset());
    CHECK(results_.Memset());
    CHECK(bool_results_.Memset());
  }

 protected:
  static gpu::GpuMemory<GF7Gpu> xs_;
  static gpu::GpuMemory<GF7Gpu> ys_;
  static gpu::GpuMemory<GF7Gpu> results_;
  static gpu::GpuMemory<bool> bool_results_;
};

gpu::GpuMemory<GF7Gpu> PrimeFieldGpuTest::xs_;
gpu::GpuMemory<GF7Gpu> PrimeFieldGpuTest::ys_;
gpu::GpuMemory<GF7Gpu> PrimeFieldGpuTest::results_;
gpu::GpuMemory<bool> PrimeFieldGpuTest::bool_results_;

}  // namespace

#define RUN_OPERATION_TESTS(method, results)                                 \
  for (size_t i = 0; i < std::size(tests); ++i) {                            \
    const auto& test = tests[i];                                             \
    xs_[i] = GF7Gpu::FromBigInt(test.x.ToBigInt());                          \
    ys_[i] = GF7Gpu::FromBigInt(test.y.ToBigInt());                          \
  }                                                                          \
  GPU_MUST_SUCCESS(                                                          \
      Launch##method(xs_.get(), ys_.get(), results.get(), std::size(tests)), \
      "Failed to " #method "()");                                            \
  for (size_t i = 0; i < std::size(tests); ++i)

#define RUN_FIELD_OPERATION_TESTS(method) \
  RUN_OPERATION_TESTS(method, results_)   \
  ASSERT_EQ(ConvertPrimeField<GF7>(results_[i]), tests[i].result)

#define RUN_COMPARISON_OPERATION_TESTS(method) \
  RUN_OPERATION_TESTS(method, bool_results_)   \
  ASSERT_EQ(bool_results_[i], tests[i].result)

TEST_F(PrimeFieldGpuTest, FromString) {
  EXPECT_EQ(*GF7Gpu::FromDecString("3"), GF7Gpu(3));
  EXPECT_FALSE(GF7Gpu::FromDecString("x").has_value());
  EXPECT_EQ(*GF7Gpu::FromHexString("0x3"), GF7Gpu(3));
  EXPECT_FALSE(GF7Gpu::FromHexString("x").has_value());
}

TEST_F(PrimeFieldGpuTest, ToString) {
  GF7Gpu f(3);

  EXPECT_EQ(f.ToString(), "3");
  EXPECT_EQ(f.ToHexString(), "0x3");
}

TEST_F(PrimeFieldGpuTest, Zero) {
  EXPECT_TRUE(GF7Gpu::Zero().IsZero());
  EXPECT_FALSE(GF7Gpu::One().IsZero());
}

TEST_F(PrimeFieldGpuTest, One) {
  EXPECT_TRUE(GF7Gpu::One().IsOneHost());
  EXPECT_FALSE(GF7Gpu::Zero().IsOneHost());
}

TEST_F(PrimeFieldGpuTest, Add) {
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

TEST_F(PrimeFieldGpuTest, Sub) {
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

TEST_F(PrimeFieldGpuTest, Mul) {
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

TEST_F(PrimeFieldGpuTest, Eq) {
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
    xs_[i] = GF7Gpu::FromBigInt(test.x.ToBigInt());
    ys_[i] = GF7Gpu::FromBigInt(test.y.ToBigInt());
    ASSERT_EQ(xs_[i] == ys_[i], test.result);
  }

  GPU_MUST_SUCCESS(
      LaunchEq(xs_.get(), ys_.get(), bool_results_.get(), std::size(tests)),
      "Failed to Eq()");
  for (size_t i = 0; i < std::size(tests); ++i) {
    ASSERT_EQ(bool_results_[i], tests[i].result);
  }
}

TEST_F(PrimeFieldGpuTest, Ne) {
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
    xs_[i] = GF7Gpu::FromBigInt(test.x.ToBigInt());
    ys_[i] = GF7Gpu::FromBigInt(test.y.ToBigInt());
    ASSERT_EQ(xs_[i] != ys_[i], test.result);
  }

  GPU_MUST_SUCCESS(
      LaunchNe(xs_.get(), ys_.get(), bool_results_.get(), std::size(tests)),
      "Failed to Ne()");
  for (size_t i = 0; i < std::size(tests); ++i) {
    ASSERT_EQ(bool_results_[i], tests[i].result);
  }
}

TEST_F(PrimeFieldGpuTest, Lt) {
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

TEST_F(PrimeFieldGpuTest, Le) {
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

TEST_F(PrimeFieldGpuTest, Gt) {
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

TEST_F(PrimeFieldGpuTest, Ge) {
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
