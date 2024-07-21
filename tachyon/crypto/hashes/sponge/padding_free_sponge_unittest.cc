#include "tachyon/crypto/hashes/sponge/padding_free_sponge.h"

#include <vector>

#include "gtest/gtest.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_horizen_external_matrix.h"
#include "tachyon/math/finite_fields/baby_bear/poseidon2.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::crypto {

using F = math::BabyBear;

namespace {

class PaddingFreeSpongeTest : public math::FiniteFieldTest<F> {};

}  // namespace

TEST_F(PaddingFreeSpongeTest, Hash) {
  using Poseidon2 = Poseidon2Sponge<
      Poseidon2ExternalMatrix<Poseidon2HorizenExternalMatrix<F>>>;
  constexpr size_t kRate = 8;
  constexpr size_t kOut = 8;

  Poseidon2Config<F> config = Poseidon2Config<F>::CreateCustom(
      15, 7, 8, 13, math::GetPoseidon2BabyBearInternalShiftVector<15>());
  Poseidon2 sponge(std::move(config));
  PaddingFreeSponge<Poseidon2, kRate, kOut> hasher(std::move(sponge));
  std::vector<F> inputs =
      base::CreateVector(100, [](uint32_t i) { return F(i); });
  std::array<F, kOut> hash = hasher.Hash(inputs);
  std::array<F, kOut> expected = {
      F(1812148253), F(1620994441), F(1186045281), F(1486390083),
      F(1521745237), F(1658565356), F(1836019216), F(3991760),
  };
  EXPECT_EQ(hash, expected);
}

}  // namespace tachyon::crypto
