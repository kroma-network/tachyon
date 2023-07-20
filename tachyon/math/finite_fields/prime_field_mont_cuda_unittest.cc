#include "gtest/gtest.h"
#include "third_party/gpus/cuda/include/cuda_runtime.h"

#include "tachyon/device/gpu/cuda/cuda_memory.h"
#include "tachyon/math/finite_fields/kernels/prime_field_ops.cu.h"
#include "tachyon/math/finite_fields/prime_field.h"

namespace tachyon {
namespace math {

namespace {

cudaError_t LaunchAdd(const GF7Cuda* x, const GF7Cuda* y, GF7Cuda* result,
                      size_t count) {
  kernels::Add<<<(count - 1) / 32 + 1, 32>>>(x, y, result, count);
  cudaError_t error = cudaGetLastError();
  GPU_LOG_IF_ERROR(ERROR, error) << "1";
  error = error ? error : cudaDeviceSynchronize();
  GPU_LOG_IF_ERROR(ERROR, error) << "2";
  return error;
}

cudaError_t LaunchSub(const GF7Cuda* x, const GF7Cuda* y, GF7Cuda* result,
                      size_t count) {
  kernels::Sub<<<(count - 1) / 32 + 1, 32>>>(x, y, result, count);
  cudaError_t error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

class PrimeFieldMontCudaTest : public testing::Test {
 public:
  constexpr static size_t N = 5;

  static void SetUpTestSuite() {
    GPU_SUCCESS(cudaDeviceReset());
    xs_ = device::gpu::MakeManagedUnique<GF7Cuda>(N * sizeof(GF7Cuda));
    ys_ = device::gpu::MakeManagedUnique<GF7Cuda>(N * sizeof(GF7Cuda));
    results_ = device::gpu::MakeManagedUnique<GF7Cuda>(N * sizeof(GF7Cuda));
    results2_ = device::gpu::MakeManagedUnique<GF7Cuda>(N * sizeof(GF7Cuda));

    GF7Config::Init();
  }

  static void TearDownTestSuite() {
    xs_.reset();
    ys_.reset();
    results_.reset();
    results2_.reset();

    GPU_SUCCESS(cudaDeviceReset());
  }

  void SetUp() override {
    GPU_SUCCESS(cudaMemset(xs_.get(), 0, N * sizeof(GF7Cuda)));
    GPU_SUCCESS(cudaMemset(ys_.get(), 0, N * sizeof(GF7Cuda)));
    GPU_SUCCESS(cudaMemset(results_.get(), 0, N * sizeof(GF7Cuda)));
    GPU_SUCCESS(cudaMemset(results2_.get(), 0, N * sizeof(GF7Cuda)));
  }

 protected:
  static device::gpu::ScopedMemory<GF7Cuda> xs_;
  static device::gpu::ScopedMemory<GF7Cuda> ys_;
  static device::gpu::ScopedMemory<GF7Cuda> results_;
  static device::gpu::ScopedMemory<GF7Cuda> results2_;
};

device::gpu::ScopedMemory<GF7Cuda> PrimeFieldMontCudaTest::xs_;
device::gpu::ScopedMemory<GF7Cuda> PrimeFieldMontCudaTest::ys_;
device::gpu::ScopedMemory<GF7Cuda> PrimeFieldMontCudaTest::results_;
device::gpu::ScopedMemory<GF7Cuda> PrimeFieldMontCudaTest::results2_;

}  // namespace

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

TEST_F(PrimeFieldMontCudaTest, AdditiveOperators) {
  struct {
    GF7 x;
    GF7 y;
    GF7 add;
    GF7 sub;
  } tests[] = {
      {GF7(3), GF7(2), GF7(5), GF7(1)},
      {GF7(2), GF7(3), GF7(5), GF7(6)},
      {GF7(5), GF7(3), GF7(1), GF7(2)},
      {GF7(3), GF7(5), GF7(1), GF7(5)},
  };

  for (size_t i = 0; i < std::size(tests); ++i) {
    const auto& test = tests[i];
    (xs_.get())[i] = GF7Cuda::FromHost(test.x);
    (ys_.get())[i] = GF7Cuda::FromHost(test.y);
  }

  GPU_SUCCESS(
      LaunchAdd(xs_.get(), ys_.get(), results_.get(), std::size(tests)));
  for (size_t i = 0; i < std::size(tests); ++i) {
    ASSERT_EQ(GF7::FromDevice((results_.get())[i]), tests[i].add);
  }

  // GPU_SUCCESS(
  //     LaunchSub(xs_.get(), ys_.get(), results2_.get(), std::size(tests)));
  // for (size_t i = 0; i < std::size(tests); ++i) {
  //   ASSERT_EQ(GF7::FromDevice((results2_.get())[i]), tests[i].sub);
  // }
}

// TEST_F(PrimeFieldMontCudaTest, DeviceXMinusZeroEqualsX) {
//   GPU_SUCCESS(LaunchSubPrimeField(xs_, zeroes_, results1_, N));
//   for (size_t i = 0; i < N; ++i) ASSERT_EQ(xs_[i], results1_[i]);
// }

// // Device x-x == 0
// TEST_F(PrimeFieldMontCudaTest, DeviceXMinusXEqualsZero) {
//   LOG(ERROR) << "!!";
//   GPU_SUCCESS(LaunchSubPrimeField(xs_, xs_, results1_, N));
//   for (size_t i = 0; i < N; ++i) ASSERT_EQ(results1_[i], zeroes_[i]);
// }

// TEST_F(PrimeFieldMontCudaTest, FromString) {
//   EXPECT_EQ(GF7::FromDecString("3"), GF7(3));
//   EXPECT_EQ(GF7::FromHexString("0x3"), GF7(3));
// }

// TEST_F(PrimeFieldMontCudaTest, ToString) {
//   GF7 f(3);

//   EXPECT_EQ(f.ToString(), "3");
//   EXPECT_EQ(f.ToHexString(), "0x3");
// }

// TEST_F(PrimeFieldMontCudaTest, Zero) {
//   EXPECT_TRUE(GF7::Zero().IsZero());
//   EXPECT_FALSE(GF7::One().IsZero());
// }

// TEST_F(PrimeFieldMontCudaTest, One) {
//   EXPECT_TRUE(GF7::One().IsOne());
//   EXPECT_FALSE(GF7::Zero().IsOne());
// }

// TEST_F(PrimeFieldMontCudaTest, AdditiveOperators) {
//   struct {
//     GF7Cuda a;
//     GF7Cuda b;
//     GF7Cuda sum;
//     GF7Cuda amb;
//     GF7Cuda bma;
//   } tests[] = {
//       {GF7Cuda(3), GF7Cuda(2), GF7Cuda(5), GF7Cuda(1),
//       GF7Cuda(6)}, {GF7Cuda(5), GF7Cuda(3), GF7Cuda(1),
//       GF7Cuda(2), GF7Cuda(5)},
//   };

//   for (const auto& test : tests) {
//     EXPECT_EQ(test.a + test.b, test.sum);
//     // EXPECT_EQ(test.b + test.a, test.sum);
//     // EXPECT_EQ(test.a - test.b, test.amb);
//     // EXPECT_EQ(test.b - test.a, test.bma);

//     // GF7Cuda tmp = test.a;
//     // tmp += test.b;
//     // EXPECT_EQ(tmp, test.sum);
//     // tmp -= test.b;
//     // EXPECT_EQ(tmp, test.a);
//   }
// }

}  // namespace math
}  // namespace tachyon
