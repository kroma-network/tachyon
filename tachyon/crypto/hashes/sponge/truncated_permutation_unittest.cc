#include "tachyon/crypto/hashes/sponge/truncated_permutation.h"

#include <vector>

#include "gtest/gtest.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_horizen_external_matrix.h"
#include "tachyon/math/finite_fields/baby_bear/poseidon2.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::crypto {

using F = math::BabyBear;
using Poseidon2 =
    Poseidon2Sponge<Poseidon2ExternalMatrix<Poseidon2HorizenExternalMatrix<F>>>;

namespace {

class TruncatedPermutationTest : public math::FiniteFieldTest<F> {};

}  // namespace

TEST_F(TruncatedPermutationTest, Hash) {
  constexpr size_t kChunk = 8;
  constexpr size_t kN = 2;

  Poseidon2Config<F> config = Poseidon2Config<F>::CreateCustom(
      15, 7, 8, 13, math::GetPoseidon2BabyBearInternalShiftVector<15>());
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
