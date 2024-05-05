// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2.h"

#include <vector>

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_horizen_external_matrix.h"
#include "tachyon/math/finite_fields/goldilocks/goldilocks.h"
#include "tachyon/math/finite_fields/goldilocks/poseidon2.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/matrix/prime_field_num_traits.h"

namespace tachyon::crypto {

using F = math::Goldilocks;

namespace {

class Poseidon2Test : public math::FiniteFieldTest<F> {};

}  // namespace

TEST_F(Poseidon2Test, Permute) {
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

TEST_F(Poseidon2Test, Copyable) {
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

}  // namespace tachyon::crypto
