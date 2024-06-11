// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#include "gtest/gtest.h"

#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_horizen_internal_matrix.h"
#include "tachyon/math/finite_fields/mersenne31/mersenne31.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/matrix/prime_field_num_traits.h"

namespace tachyon::crypto {

using F = math::Mersenne31;

namespace {

class Poseidon2InternalMatrixTest : public math::FiniteFieldTest<F> {};

}  // namespace

TEST_F(Poseidon2InternalMatrixTest, ApplyHorizen) {
  math::Vector<F> diagonal_minus_one_vec{
      {F::Random() - F::One(), F::Random() - F::One(), F::Random() - F::One()}};

  math::Matrix<F> matrix{
      {diagonal_minus_one_vec[0] + F::One(), F::One(), F::One()},
      {F::One(), diagonal_minus_one_vec[1] + F::One(), F::One()},
      {F::One(), F::One(), diagonal_minus_one_vec[2] + F::One()},
  };

  math::Vector<F> state{{F::Random(), F::Random(), F::Random()}};
  math::Vector<F> state2 = state;
  Poseidon2HorizenInternalMatrix<F>::Apply(state2, diagonal_minus_one_vec);
  EXPECT_EQ(matrix * state, state2);
}

}  // namespace tachyon::crypto
