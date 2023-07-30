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

device::gpu::ScopedMemory<GF7Cuda> PrimeFieldCudaTest::xs_;
device::gpu::ScopedMemory<GF7Cuda> PrimeFieldCudaTest::ys_;
device::gpu::ScopedMemory<GF7Cuda> PrimeFieldCudaTest::results_;
device::gpu::ScopedMemory<bool> PrimeFieldCudaTest::bool_results_;

}  // namespace
}  // namespace tachyon::math
