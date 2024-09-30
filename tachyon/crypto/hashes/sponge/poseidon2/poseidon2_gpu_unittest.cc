#include <vector>

#include "gtest/gtest.h"

#include "tachyon/crypto/hashes/sponge/poseidon2/icicle/icicle_poseidon2_holder.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/param_traits/poseidon2_baby_bear.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/param_traits/poseidon2_bn254.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_params.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::crypto {

namespace {

template <typename Params>
class Poseidon2GpuTest : public math::FiniteFieldTest<typename Params::Field> {
};

}  // namespace

using ParamsTypes = testing::Types<
    Poseidon2Params<Poseidon2Vendor::kPlonky3, Poseidon2Vendor::kPlonky3,
                    math::BabyBear, 15, 7>,
    Poseidon2Params<Poseidon2Vendor::kHorizen, Poseidon2Vendor::kHorizen,
                    math::bn254::Fr, 2, 5>>;
TYPED_TEST_SUITE(Poseidon2GpuTest, ParamsTypes);

TYPED_TEST(Poseidon2GpuTest, Poseidon2Correctness) {
  using Params = TypeParam;
  using F = typename Params::Field;

  constexpr size_t kLen = 16;
  constexpr size_t kRate = std::is_same_v<F, math::BabyBear> ? 8 : 2;
  constexpr size_t kOutput = std::is_same_v<F, math::BabyBear> ? 6 : 1;

  Poseidon2Config<Params> config = Poseidon2Config<Params>::CreateDefault();
  IciclePoseidon2Holder<F> holder =
      IciclePoseidon2Holder<F>::template Create<Params::kWidth - 1>(config);

  std::vector<F> inputs =
      base::CreateVector(kLen * kRate, []() { return F::Random(); });
  std::vector<F> expected;
  expected.reserve(kLen * kOutput);

  Poseidon2Sponge<Params> poseidon2_cpu(config);
  for (size_t i = 0; i < kLen; ++i) {
    SpongeState<Params> state;
    for (size_t j = 0; j < kRate; ++j) {
      state[j] = inputs[i * kRate + j];
    }
    poseidon2_cpu.Permute(state);
    for (size_t j = 0; j < kOutput; ++j) {
      expected.push_back(state.elements[j]);
    }
  }

  std::vector<F> outputs(kLen * kOutput);
  ASSERT_TRUE((*holder).Hash(kRate, inputs, absl::MakeSpan(outputs)));
  EXPECT_EQ(expected, outputs);
}

}  // namespace tachyon::crypto
