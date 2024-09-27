#include "tachyon/crypto/hashes/sponge/truncated_permutation.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/param_traits/poseidon2_baby_bear.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_params.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::crypto {

using F = math::BabyBear;
using Params = Poseidon2Params<Poseidon2Vendor::kHorizen,
                               Poseidon2Vendor::kPlonky3, F, 15, 7>;
using Poseidon2 = Poseidon2Sponge<Params>;

namespace {

class TruncatedPermutationTest : public math::FiniteFieldTest<F> {};

}  // namespace

TEST_F(TruncatedPermutationTest, Hash) {
  constexpr size_t kChunk = 8;
  constexpr size_t kN = 2;

  auto config =
      Poseidon2Config<Params>::Create(GetPoseidon2InternalShiftArray<Params>());
  Poseidon2 sponge(std::move(config));
  TruncatedPermutation<Poseidon2, kChunk, kN> compressor(std::move(sponge));
  std::vector<std::vector<F>> inputs = base::CreateVector(kN, [](uint32_t i) {
    return base::CreateVector(kChunk,
                              [i](uint32_t j) { return F(i * kChunk + j); });
  });
  std::array<F, kChunk> hash = compressor.Compress(inputs);
  std::array<F, kChunk> expected = {F(1699737005), F(296394369),  F(268410240),
                                    F(828329642),  F(1491697358), F(1128780676),
                                    F(287184043),  F(1806152977)};
  EXPECT_EQ(hash, expected);
}

}  // namespace tachyon::crypto
