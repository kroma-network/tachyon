// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2.h"

#include <vector>

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_horizen_external_matrix.h"
#include "tachyon/math/finite_fields/baby_bear/packed_baby_bear.h"
#include "tachyon/math/finite_fields/baby_bear/poseidon2.h"
#include "tachyon/math/finite_fields/goldilocks/poseidon2.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/matrix/prime_field_num_traits.h"

namespace tachyon::crypto {

namespace {

class Poseidon2GoldilocksTest : public math::FiniteFieldTest<math::Goldilocks> {
};

}  // namespace

TEST_F(Poseidon2GoldilocksTest, Permute) {
  using F = math::Goldilocks;

  Poseidon2Config<F> config = Poseidon2Config<F>::CreateCustom(
      7, 7, 8, 22, math::GetPoseidon2GoldilocksInternalDiagonalVector<8>());
  Poseidon2Sponge<Poseidon2ExternalMatrix<Poseidon2HorizenExternalMatrix<F>>>
      sponge(config);
  for (size_t i = 0; i < 8; ++i) {
    sponge.state.elements[i] = F(i);
  }
  sponge.Permute();
  math::Vector<F> expected{
      {F(UINT64_C(14266028122062624699))}, {F(UINT64_C(5353147180106052723))},
      {F(UINT64_C(15203350112844181434))}, {F(UINT64_C(17630919042639565165))},
      {F(UINT64_C(16601551015858213987))}, {F(UINT64_C(10184091939013874068))},
      {F(UINT64_C(16774100645754596496))}, {F(UINT64_C(12047415603622314780))},
  };
  EXPECT_EQ(sponge.state.elements, expected);
}

TEST_F(Poseidon2GoldilocksTest, Copyable) {
  using F = math::Goldilocks;

  Poseidon2Config<F> config = Poseidon2Config<F>::CreateCustom(
      7, 7, 8, 22, math::GetPoseidon2GoldilocksInternalDiagonalVector<8>());
  Poseidon2Sponge<Poseidon2ExternalMatrix<Poseidon2HorizenExternalMatrix<F>>>
      expected(config);

  base::Uint8VectorBuffer write_buf;
  ASSERT_TRUE(write_buf.Grow(base::EstimateSize(expected)));
  ASSERT_TRUE(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  Poseidon2Sponge<Poseidon2ExternalMatrix<Poseidon2HorizenExternalMatrix<F>>>
      value;
  ASSERT_TRUE(write_buf.Read(&value));

  EXPECT_EQ(value, expected);
}

namespace {

class Poseidon2BabyBearTest
    : public math::FiniteFieldTest<math::PackedBabyBear> {};

}  // namespace

TEST_F(Poseidon2BabyBearTest, Permute) {
  using F = math::BabyBear;

  Poseidon2Config<F> config = Poseidon2Config<F>::CreateCustom(
      15, 7, 8, 13, math::GetPoseidon2BabyBearInternalShiftVector<15>());
  Poseidon2Sponge<Poseidon2ExternalMatrix<Poseidon2HorizenExternalMatrix<F>>>
      sponge(config);
  for (size_t i = 0; i < 16; ++i) {
    sponge.state.elements[i] = F(i);
  }
  sponge.Permute();
  math::Vector<F> expected{
      {F(1699737005)}, {F(296394369)},  {F(268410240)},  {F(828329642)},
      {F(1491697358)}, {F(1128780676)}, {F(287184043)},  {F(1806152977)},
      {F(1380147856)}, {F(345666717)},  {F(491196631)},  {F(1875224538)},
      {F(697740550)},  {F(1854502887)}, {F(1201727753)}, {F(1802410886)},
  };
  EXPECT_EQ(sponge.state.elements, expected);
}

TEST_F(Poseidon2BabyBearTest, PermutePacked) {
  using PackedF = math::PackedBabyBear;
  using F = math::BabyBear;

  Poseidon2Config<PackedF> packed_config =
      Poseidon2Config<PackedF>::CreateCustom(
          15, 7, 8, 13, math::GetPoseidon2BabyBearInternalShiftVector<15>());
  Poseidon2Sponge<
      Poseidon2ExternalMatrix<Poseidon2HorizenExternalMatrix<PackedF>>>
      packed_sponge(packed_config);
  for (size_t i = 0; i < 16; ++i) {
    packed_sponge.state.elements[i] = PackedF(i);
  }
  packed_sponge.Permute();

  Poseidon2Config<F> config = Poseidon2Config<F>::CreateCustom(
      15, 7, 8, 13, math::GetPoseidon2BabyBearInternalShiftVector<15>());
  Poseidon2Sponge<Poseidon2ExternalMatrix<Poseidon2HorizenExternalMatrix<F>>>
      sponge(config);
  for (size_t i = 0; i < 16; ++i) {
    sponge.state.elements[i] = F(i);
  }
  sponge.Permute();

  for (size_t i = 0; i < 16; ++i) {
    EXPECT_EQ(packed_sponge.state.elements[i],
              PackedF::Broadcast(sponge.state.elements[i]));
  }
}

}  // namespace tachyon::crypto
