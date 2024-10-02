// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#include "gtest/gtest.h"

#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_horizen_internal_matrix.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_plonky3_internal_matrix.h"
#include "tachyon/math/finite_fields/mersenne31/mersenne31.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/matrix/matrix_utils.h"

namespace tachyon::crypto {

namespace {

template <typename F>
class Poseidon2InternalMatrixTest : public math::FiniteFieldTest<F> {};

}  // namespace

using FieldTypes = testing::Types<math::Mersenne31, math::PackedMersenne31>;
TYPED_TEST_SUITE(Poseidon2InternalMatrixTest, FieldTypes);

TYPED_TEST(Poseidon2InternalMatrixTest, ApplyHorizen) {
  using F = TypeParam;

  math::Vector<F, 3> diagonal_minus_one_vec =
      math::Vector<F, 3>::Random() - math::Vector<F, 3>::Constant(F::One());

  math::Matrix<F, 3, 3> matrix{
      {diagonal_minus_one_vec[0] + F::One(), F::One(), F::One()},
      {F::One(), diagonal_minus_one_vec[1] + F::One(), F::One()},
      {F::One(), F::One(), diagonal_minus_one_vec[2] + F::One()},
  };

  math::Vector<F, 3> state = math::Vector<F, 3>::Random();
  std::array<F, 3> state2 = math::ToArray(state);
  Poseidon2HorizenInternalMatrix<F>::Apply(state2, diagonal_minus_one_vec);
  EXPECT_EQ(math::ToArray(matrix * state), state2);
}

}  // namespace tachyon::crypto
