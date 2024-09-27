// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/param_traits/poseidon2_baby_bear.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/param_traits/poseidon2_goldilocks.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_params.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::crypto {

namespace {

class Poseidon2GoldilocksTest : public math::FiniteFieldTest<math::Goldilocks> {
};

}  // namespace

TEST_F(Poseidon2GoldilocksTest, Permute) {
  using F = math::Goldilocks;
  using Params = Poseidon2Params<Poseidon2Vendor::kHorizen,
                                 Poseidon2Vendor::kPlonky3, F, 7, 7>;

  auto config = Poseidon2Config<Params>::Create(
      crypto::GetPoseidon2InternalDiagonalArray<Params>());
  Poseidon2Sponge<Params> sponge(std::move(config));
  SpongeState<Params> state;
  for (size_t i = 0; i < 8; ++i) {
    state.elements[i] = F(i);
  }
  sponge.Permute(state);
  math::Vector<F> expected{
      {F(UINT64_C(14266028122062624699))}, {F(UINT64_C(5353147180106052723))},
      {F(UINT64_C(15203350112844181434))}, {F(UINT64_C(17630919042639565165))},
      {F(UINT64_C(16601551015858213987))}, {F(UINT64_C(10184091939013874068))},
      {F(UINT64_C(16774100645754596496))}, {F(UINT64_C(12047415603622314780))},
  };
  EXPECT_EQ(state.elements, expected);
}

TEST_F(Poseidon2GoldilocksTest, Copyable) {
  using F = math::Goldilocks;
  using Params = Poseidon2Params<Poseidon2Vendor::kHorizen,
                                 Poseidon2Vendor::kPlonky3, F, 7, 7>;

  auto config = Poseidon2Config<Params>::Create(
      crypto::GetPoseidon2InternalDiagonalArray<Params>());
  Poseidon2Sponge<Params> expected(std::move(config));

  base::Uint8VectorBuffer write_buf;
  ASSERT_TRUE(write_buf.Grow(base::EstimateSize(expected)));
  ASSERT_TRUE(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  Poseidon2Sponge<Params> value;
  ASSERT_TRUE(write_buf.Read(&value));

  EXPECT_EQ(value, expected);
}

namespace {

class Poseidon2BabyBearTest
    : public math::FiniteFieldTest<math::PackedBabyBear> {};

}  // namespace

TEST_F(Poseidon2BabyBearTest, Permute) {
  using F = math::BabyBear;
  using Params = Poseidon2Params<Poseidon2Vendor::kHorizen,
                                 Poseidon2Vendor::kPlonky3, F, 15, 7>;

  auto config =
      Poseidon2Config<Params>::Create(GetPoseidon2InternalShiftArray<Params>());
  Poseidon2Sponge<Params> sponge(std::move(config));
  SpongeState<Params> state;
  for (size_t i = 0; i < 16; ++i) {
    state.elements[i] = F(i);
  }
  sponge.Permute(state);
  math::Vector<F> expected{
      {F(1699737005)}, {F(296394369)},  {F(268410240)},  {F(828329642)},
      {F(1491697358)}, {F(1128780676)}, {F(287184043)},  {F(1806152977)},
      {F(1380147856)}, {F(345666717)},  {F(491196631)},  {F(1875224538)},
      {F(697740550)},  {F(1854502887)}, {F(1201727753)}, {F(1802410886)},
  };
  EXPECT_EQ(state.elements, expected);
}

TEST_F(Poseidon2BabyBearTest, PermutePacked) {
  using F = math::BabyBear;
  using PackedF = math::PackedBabyBear;
  using Params = Poseidon2Params<Poseidon2Vendor::kHorizen,
                                 Poseidon2Vendor::kPlonky3, F, 15, 7>;
  using PackedParams =
      Poseidon2Params<Poseidon2Vendor::kHorizen, Poseidon2Vendor::kPlonky3,
                      PackedF, 15, 7>;

  auto packed_config = Poseidon2Config<PackedParams>::Create(
      GetPoseidon2InternalShiftArray<PackedParams>());
  Poseidon2Sponge<PackedParams> packed_sponge(std::move(packed_config));
  SpongeState<PackedParams> packed_state;
  for (size_t i = 0; i < 16; ++i) {
    packed_state.elements[i] = PackedF(i);
  }
  packed_sponge.Permute(packed_state);

  auto config =
      Poseidon2Config<Params>::Create(GetPoseidon2InternalShiftArray<Params>());
  Poseidon2Sponge<Params> sponge(std::move(config));
  SpongeState<Params> state;
  for (size_t i = 0; i < 16; ++i) {
    state.elements[i] = F(i);
  }
  sponge.Permute(state);

  for (size_t i = 0; i < 16; ++i) {
    EXPECT_EQ(packed_state.elements[i], PackedF::Broadcast(state.elements[i]));
  }
}

}  // namespace tachyon::crypto
