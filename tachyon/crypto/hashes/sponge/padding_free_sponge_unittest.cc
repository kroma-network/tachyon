#include "tachyon/crypto/hashes/sponge/padding_free_sponge.h"

#include <vector>

#include "gtest/gtest.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/param_traits/poseidon2_baby_bear.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_params.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::crypto {

using F = math::BabyBear;

namespace {

class PaddingFreeSpongeTest : public math::FiniteFieldTest<F> {};

}  // namespace

TEST_F(PaddingFreeSpongeTest, Hash) {
  using Params = Poseidon2Params<Poseidon2Vendor::kHorizen,
                                 Poseidon2Vendor::kPlonky3, F, 15, 7>;
  using Poseidon2 = Poseidon2Sponge<Params>;
  constexpr size_t kRate = 8;
  constexpr size_t kOut = 8;

  Poseidon2 sponge;
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
